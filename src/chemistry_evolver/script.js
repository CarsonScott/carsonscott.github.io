// ============================================================================
// sim.js — Spatial-Chemical System model (pure, environment-agnostic)
//
// No p5 / DOM dependency. Safe to load in a browser (attaches `Sim` to the
// global scope) or in Node (via module.exports). Mirrors system_model.md.
//
// State: graph G=(N,E); node n holds channel vector c_n in R_+^C.
// One tick moves fraction flowRate of each channel along a temp-softmax over
// neighbor scores  s = F * c_v - crowd * occupancy_v.  By default mass is
// conserved (closed). The Interface layer (system_outline.txt §IV) opens the
// system: nodes flagged `isInterface` act as sources/sinks and, when a parent
// Sim is attached, as conduits translating chemical flow to a global parent.
// ============================================================================

class Sim {
  constructor({ channels = 3, size = 120 } = {}) {
    this.C = channels;                       // number of chemical channels
    this.OFFSETS = [[-1, 0], [0, -1], [0, 1], [1, 0]]; // von Neumann, toroidal
    this.params = {
      capacity: 2.0,
      flowRate: 0.15,
      temp: 0.30,
      crowd: 0.50,
      conduitRate: 0.5,
    };
    this.F = [];        // C x C affinity kernel
    this.setSize(size);
  }

  // (re)allocate buffers for a grid of side `w`; keeps or rolls forces.
  setSize(w) {
    this.W = w;
    this.stores = new Float32Array(this.W * this.W * this.C);
    this.next = new Float32Array(this.W * this.W * this.C);
    this.occ = new Float32Array(this.W * this.W);
    this._allocInterface();
    this._seedGrid();
  }

  // (re)allocate interface-layer state for the current grid.
  _allocInterface() {
    const n = this.W * this.W;
    // node-role mask: 1 = interface point (system_outline.txt §IV)
    this.isInterface = new Uint8Array(n);
    // direction tag per interface point: -1 = incoming (absorb from parent),
    // +1 = outgoing (expel to parent). 0 = not an interface point. Enables
    // mass conservation between the two systems.
    this.interfaceDir = new Int8Array(n);
    // serializable list of interface points {i, j, dir} (the genome's interface
    // pattern). Seeds are generated randomly (see randomizeInterfaces) and are
    // the source of truth for isInterface/interfaceDir.
    this.interfacePatches = [];
    // per-channel global source/sink rates applied at every interface node
    this.source = new Float32Array(this.C);
    this.sink = new Float32Array(this.C);
    // optional recursive coupling to a global parent system (§IV/§V conduit)
    this.parent = null;        // parent Sim, or null (closed)
    this.parentMap = null;     // Int array length C: local channel t -> parent channel
    // continuous global position of this system's origin in parent space
    this.globalPos = [0, 0];
    // bin size: scaling factor projecting this system onto the parent grid
    // (parent cell = floor((local + globalPos) / binSize)). 1 = aligned grids.
    this.binSize = 1;
  }

  // randomize the affinity kernel ("species relationships")
  randomizeForces() {
    const rnd = () => (Math.random() < 0.5 ? (Math.random() * 2 - 1) : 0);
    this.F = Array.from({ length: this.C }, () =>
      Array.from({ length: this.C }, rnd));
  }

  _seedGrid() {
    for (let i = 0; i < this.stores.length; i++) this.stores[i] = Math.random() * 0.5;
  }

  // fresh field at current size, preserving the affinity kernel
  rebuildGrid() { this._seedGrid(); }

  // ---- Interface layer (system_outline.txt §IV) -------------------------
  // Designate specific nodes as interface points. Accepts:
  //   setInterface(list)                       — list/predicate, dir tagged per-point
  //   setInterface(list, dirFn)                — dirFn(i,j) => -1 | +1 per point
  // where `list` is an array of [i,j] pairs, flat indices, or a predicate.
  // Direction: -1 incoming (absorb), +1 outgoing (expel). Default tag alternates
  // by (i+j) parity so each subsystem gets a balanced in/out interface.
  setInterface(list, dirFn = null) {
    this.clearInterface();
    if (typeof list === 'function') {
      for (let i = 0; i < this.W; i++)
        for (let j = 0; j < this.W; j++)
          if (list(i, j)) this._markInterface(i, j, dirFn);
      return;
    }
    for (const item of list) {
      let i, j;
      if (Array.isArray(item)) { [i, j] = item; }
      else { const idx = item; i = Math.floor(idx / this.W); j = idx % this.W; }
      if (i >= 0 && i < this.W && j >= 0 && j < this.W) this._markInterface(i, j, dirFn);
    }
  }

  _markInterface(i, j, dirFn) {
    const k = i * this.W + j;
    this.isInterface[k] = 1;
    this.interfaceDir[k] = dirFn ? dirFn(i, j) : ((i + j) % 2 === 0 ? -1 : 1);
    this.interfacePatches.push({ i, j, dir: this.interfaceDir[k] });
  }

  // Apply a genome-defined interface pattern: an array of {i, j, dir} where
  // dir is -1 (incoming) or +1 (outgoing). Replaces any existing interfaces.
  // This is the canonical, serializable representation of a subsystem's
  // interface points (randomly distributed within the subspace per genome).
  applyInterfacePatches(patches) {
    this.clearInterface();
    for (const p of patches || []) {
      const i = p.i | 0, j = p.j | 0;
      if (i < 0 || i >= this.W || j < 0 || j >= this.W) continue;
      const dir = p.dir < 0 ? -1 : 1;
      const k = i * this.W + j;
      this.isInterface[k] = 1;
      this.interfaceDir[k] = dir;
      this.interfacePatches.push({ i, j, dir });
    }
  }

  // Generate a random interface pattern of `count` points scattered anywhere
  // inside the subspace (not just the border ring), with balanced in/out
  // directions so coupling stays conservative. Stored as the genome interface
  // pattern and returned for serialization.
  randomizeInterfaces(count) {
    const cs = this.W;
    const n = count || Math.max(4, Math.floor(cs * cs * 0.12));
    const patches = [];
    const used = new Set();
    let incoming = 0, outgoing = 0;
    let guard = n * 40;
    while (patches.length < n && guard-- > 0) {
      const i = Math.floor(Math.random() * cs);
      const j = Math.floor(Math.random() * cs);
      const key = i * cs + j;
      if (used.has(key)) continue;
      used.add(key);
      // alternate direction to keep in/out balanced for conservation
      const dir = incoming <= outgoing ? -1 : 1;
      if (dir < 0) incoming++; else outgoing++;
      patches.push({ i, j, dir });
    }
    this.applyInterfacePatches(patches);
    return patches;
  }

  // Clear all interface designations (returns system to fully closed state).
  clearInterface() {
    if (this.isInterface) this.isInterface.fill(0);
    if (this.interfaceDir) this.interfaceDir.fill(0);
    this.interfacePatches = [];
  }

  // Per-channel global source/sink rates applied at every interface node each
  // tick. `arr` length C (or scalar broadcast). Zero = no open flux.
  setSourceRate(arr) {
    const a = typeof arr === 'number' ? new Array(this.C).fill(arr) : arr;
    for (let t = 0; t < this.C; t++) this.source[t] = Math.max(0, +a[t] || 0);
  }
  setSinkRate(arr) {
    const a = typeof arr === 'number' ? new Array(this.C).fill(arr) : arr;
    for (let t = 0; t < this.C; t++) this.sink[t] = Math.max(0, +a[t] || 0);
  }

  // Attach a global parent system. The child is PROJECTED onto the parent grid
  // via `binSize`: a parent cell = floor((localIdx + globalPos) / binSize).
  // `globalPos` is the child's continuous origin in parent space; `channelMap`
  // maps local channel t -> parent channel. Detach with detachParent().
  // This is the §IV/§V "global position" conduit with directional interface
  // points (see interfaceDir) for conservation.
  attachParent(parentSim, channelMap, globalPos = [0, 0], binSize = 1) {
    this.parent = parentSim;
    this.parentMap = channelMap && channelMap.length === this.C
      ? channelMap.slice() : Array.from({ length: this.C }, (_, t) => t);
    this.globalPos = [globalPos[0] || 0, globalPos[1] || 0];
    this.binSize = binSize > 0 ? binSize : 1;
  }
  detachParent() { this.parent = null; this.parentMap = null; this.globalPos = [0, 0]; this.binSize = 1; }

  // Map a local interface cell (i,j) to its parent grid coordinate (bin) using
  // the current globalPos + binSize projection. globalPos is the CONTINUOUS
  // CENTER of the subspace in parent space (units: parentCell * binSize). The
  // subspace is centered on globalPos: local cell (i,j) sits at offset
  // (j - (cs-1)/2, i - (cs-1)/2) from center, so x uses j and y uses i (no
  // accidental axis swap). The parent space is TOROIDAL: projected coords wrap
  // modulo parent.W so subsystems can drift off one edge and reappear on the
  // other. Returns the wrapped [pi, pj].
  interfaceParentCoord(i, j) {
    if (!this.parent) return null;
    const cs = this.W;
    const PW = this.parent.W;
    const off = (cs - 1) / 2;
    let pi = Math.floor((this.globalPos[0] + (j - off) * this.binSize) / this.binSize);
    let pj = Math.floor((this.globalPos[1] + (i - off) * this.binSize) / this.binSize);
    pi = ((pi % PW) + PW) % PW;
    pj = ((pj % PW) + PW) % PW;
    return [pi, pj];
  }

  // Directional electrochemical potential at interface point (i,j): the
  // gradient of the chemical vector between this point and the parent bin
  // directly below it, plus influence from the parent's surrounding
  // neighborhood. Returns a per-channel delta [C] used to bias movement:
  // align (attraction) when the local vector matches the parent's, repel when
  // they oppose. Magnitude scales with the channel difference.
  interfacePotential(i, j) {
    const grad = new Array(this.C).fill(0);
    if (!this.parent) return grad;
    const coord = this.interfaceParentCoord(i, j);
    if (!coord) return grad;
    const [pi, pj] = coord;
    const p = this.parent;
    const localBase = this._idx(i, j, 0);
    // local chemical vector at this interface point
    const lv = [];
    for (let t = 0; t < this.C; t++) lv.push(this.stores[localBase + t]);
    // neighborhood influence: average of the 3x3 parent bins around the target
    let sum = 0, n = 0;
    const nb = [];
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
      const ni = pi + a, nj = pj + b;
      if (ni < 0 || ni >= p.W || nj < 0 || nj >= p.W) continue;
      const v = [];
      for (let t = 0; t < this.C; t++) v.push(p.stores[p._idx(ni, nj, t)]);
      nb.push(v); sum++; n++;
    }
    const meanV = new Array(this.C).fill(0);
    for (const v of nb) for (let t = 0; t < this.C; t++) meanV[t] += v[t] / n;
    // gradient: how much the parent (neighborhood) pulls the local vector
    for (let t = 0; t < this.C; t++) grad[t] = meanV[t] - lv[t];
    return grad;
  }

  // full reset: new forces + new field
  regenerate() { this.randomizeForces(); this._seedGrid(); }

  _idx(i, j, t) { return (i * this.W + j) * this.C + t; }
  _mod(v, a) { return ((v % a) + a) % a; }

  // advance the system by one tick (in place, double-buffered)
  step() {
    const { W, C, stores, next, occ, F, OFFSETS, params } = this;
    const { capacity, flowRate, temp, crowd } = params;

    // occupancy: node-local scalar field o_n = sum_t c_n[t] / capacity
    for (let i = 0; i < W; i++) {
      for (let j = 0; j < W; j++) {
        let s = 0;
        for (let t = 0; t < C; t++) s += stores[this._idx(i, j, t)];
        occ[i * W + j] = s / capacity;
      }
    }

    next.set(stores);
    const sc = new Array(OFFSETS.length);

    for (let i = 0; i < W; i++) {
      for (let j = 0; j < W; j++) {
        const nb = OFFSETS.map(([a, b]) => [this._mod(i + a, W), this._mod(j + b, W)]);
        for (let t = 0; t < C; t++) {
          const amt = stores[this._idx(i, j, t)];
          if (amt <= 0) continue;

          // score each neighbor: affinity pull minus crowding repulsion
          let mx = -Infinity;
          for (let k = 0; k < nb.length; k++) {
            const [ni, nj] = nb[k];
            let s = -crowd * occ[ni * W + nj];
            for (let jt = 0; jt < C; jt++) s += F[t][jt] * stores[this._idx(ni, nj, jt)];
            sc[k] = s;
            if (s > mx) mx = s;
          }

          // temperature-controlled softmax → routing distribution
          let se = 0;
          for (let k = 0; k < nb.length; k++) { sc[k] = Math.exp((sc[k] - mx) / temp); se += sc[k]; }

          // conservative flux: move flowRate * amt out, redistributed by p
          const out = flowRate * amt;
          next[this._idx(i, j, t)] -= out;
          for (let k = 0; k < nb.length; k++) {
            const [ni, nj] = nb[k];
            next[this._idx(ni, nj, t)] += out * (sc[k] / se);
          }
        }
      }
    }

    // ---- Interface layer (§IV): open the system at interface nodes ----
    // Source (inject) and sink (extract) are applied on `next` so they enter
    // the new state this tick. With zero source/sink and no parent the system
    // remains exactly conservative (edge flux block above is unchanged).
    let nSrc = 0, nSink = 0;
    for (let i = 0; i < W; i++) {
      for (let j = 0; j < W; j++) {
        if (!this.isInterface[i * W + j]) continue;
        const base = this._idx(i, j, 0);
        for (let t = 0; t < C; t++) {
          if (this.source[t] > 0) { next[base + t] += this.source[t]; nSrc++; }
          if (this.sink[t] > 0) {
            const out = Math.min(this.sink[t], next[base + t]);
            next[base + t] -= out; nSink++;
          }
        }
      }
    }

    // Parent conduit (§IV/§V): each DIRECTIONAL interface point is projected
    // onto the parent grid via binSize and reads/writes the parent bin directly
    // below it. Incoming points (-1) ABSORB a conduitRate fraction of the
    // parent bin's mass; outgoing points (+1) EXPEL a conduitRate fraction of
    // their own mass into the parent bin. Local->parent writes to p.stores
    // immediately; parent->local is applied on `next`. Directional tags keep
    // the exchange conservative (every expel is matched by an absorb elsewhere).
    if (this.parent) {
      const p = this.parent;
      if (p.stores) {
        const frac = params.conduitRate != null ? params.conduitRate : flowRate;
        for (let i = 0; i < W; i++) {
          for (let j = 0; j < W; j++) {
            const k = i * W + j;
            if (!this.isInterface[k]) continue;
            const coord = this.interfaceParentCoord(i, j);
            if (!coord) continue;
            const [pi, pj] = coord;
            const dir = this.interfaceDir[k];
            for (let t = 0; t < C; t++) {
              const pt = this.parentMap[t];
              if (pt < 0 || pt >= p.C) continue;
              const local = this._idx(i, j, t);
              const par = p._idx(pi, pj, pt);
              if (dir < 0) {
                // incoming: pull parent mass into this interface point
                const inAmt = frac * p.stores[par];
                next[local] += inAmt; p.stores[par] -= inAmt;
              } else if (dir > 0) {
                // outgoing: expel this interface point's mass into the parent
                const outAmt = frac * stores[local];
                next[local] -= outAmt; p.stores[par] += outAmt;
              }
            }
          }
        }
      }
    }

    const tmp = this.stores; this.stores = this.next; this.next = tmp;
  }

  // build a Sim directly from a genome (F kernel + params), skipping re-roll
  static fromGenome(genome, size) {
    const sim = new Sim({ channels: genome.F.length, size });
    sim.F = genome.F.map((row) => row.slice());
    Object.assign(sim.params, genome.params);
    sim._seedGrid();
    return sim;
  }

  // overwrite this sim's rules from a genome (keeps current buffers/size)
  applyGenome(genome) {
    this.C = genome.F.length;
    this.F = genome.F.map((row) => row.slice());
    Object.assign(this.params, genome.params);
    if (Array.isArray(genome.interfaces)) this.applyInterfacePatches(genome.interfaces);
  }

  // current rules as a plain genome object (for breeding / serialization)
  toGenome() {
    return {
      F: this.F.map((row) => row.slice()),
      params: Object.assign({}, this.params),
      interfaces: this.interfacePatches.map((p) => ({ i: p.i, j: p.j, dir: p.dir })),
    };
  }

  // ---- analysis helpers ---------------------------------------------------
  // total mass per channel (should be constant under conservation)
  channelTotals() {
    const tot = new Array(this.C).fill(0);
    for (let i = 0; i < this.W; i++)
      for (let j = 0; j < this.W; j++)
        for (let t = 0; t < this.C; t++) tot[t] += this.stores[this._idx(i, j, t)];
    return tot;
  }

  // per-channel mean and std across the grid (spatial distribution stats)
  channelStats() {
    const stats = [];
    for (let t = 0; t < this.C; t++) {
      let sum = 0, sumSq = 0, n = this.W * this.W;
      for (let i = 0; i < this.W; i++)
        for (let j = 0; j < this.W; j++) {
          const v = this.stores[this._idx(i, j, t)];
          sum += v; sumSq += v * v;
        }
      const mean = sum / n;
      const variance = Math.max(0, sumSq / n - mean * mean);
      stats.push({ mean, std: Math.sqrt(variance) });
    }
    return stats;
  }

  // full grid snapshot as a nested array [i][j][t] (for agent consumption)
  snapshot() {
    const grid = [];
    for (let i = 0; i < this.W; i++) {
      const row = [];
      for (let j = 0; j < this.W; j++) {
        const cell = [];
        for (let t = 0; t < this.C; t++) cell.push(this.stores[this._idx(i, j, t)]);
        row.push(cell);
      }
      grid.push(row);
    }
    return grid;
  }
}

// Support both browser (global) and Node (module) environments.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Sim };
} else {
  globalThis.Sim = Sim;
}

// ============================================================================
// evolve.js — interactive evolutionary selector (human-in-the-loop)
//
// A population of Spatial-Chemical Systems runs live, side by side. The user
// watches them evolve and CLICKS winners. "evolve" breeds the next generation:
//   next = [exact copies of each winner] + [mutated copies of winners]
//          to fill the population; losers are discarded. Winners always survive
//          via their straight copies. No fitness function — the human selects.
//
// Genome = { F: number[C][C] (affinity kernel), params: {flowRate,temp,capacity,crowd} }
// ============================================================================

const PARAM_RANGES = {
  flowRate: [0, 1],
  temp: [0.02, 2],
  capacity: [0.2, 10],
  crowd: [0, 3],
};

function clamp(v, [lo, hi]) { return Math.min(hi, Math.max(lo, v)); }
function gauss() { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function randForce() { return Math.random() < 0.5 ? (Math.random() * 2 - 1) : 0; }

function randomGenome(C = 3) {
  const F = Array.from({ length: C }, () =>
    Array.from({ length: C }, randForce));
  const params = {
    flowRate: Math.random(),
    temp: 0.02 + Math.random() * 1.98,
    capacity: 0.2 + Math.random() * 9.8,
    crowd: Math.random() * 3,
  };
  return { F, params };
}

// mutate a genome, returning a new one (parent untouched)
function mutateGenome(g, rate) {
  const F = g.F.map((row) => row.map((x) => {
    let v = x;
    if (Math.random() < rate) v += gauss() * 0.25;          // perturb
    if (Math.random() < rate * 0.3) v = randForce();        // occasional hard flip
    return Math.max(-3, Math.min(3, v));
  }));
  const params = {};
  for (const k of Object.keys(PARAM_RANGES)) {
    let v = g.params[k];
    if (Math.random() < rate) v = clamp(v + gauss() * 0.15, PARAM_RANGES[k]);
    params[k] = v;
  }
  return { F, params };
}

// crossover two parent genomes into a child (uniform per-element/per-param)
function crossoverGenomes(a, b) {
  const F = a.F.map((row, i) => row.map((x, j) => {
    // 50/50 pick of each parent's element, with a slight blend option
    const pick = Math.random();
    if (pick < 0.45) return x;            // from A
    if (pick < 0.90) return b.F[i][j];    // from B
    return (x + b.F[i][j]) / 2;           // blend
  }));
  const params = {};
  for (const k of Object.keys(PARAM_RANGES)) {
    params[k] = Math.random() < 0.5 ? a.params[k] : b.params[k];
  }
  return { F, params };
}

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------
class Population {
  constructor(n, size, mutRate, speed) {
    this.n = n;
    this.size = size;
    this.mutRate = mutRate;
    this.speed = speed;
    this.generation = 0;
    this.lastWinners = null;
    this.crossoverRate = 0.5;
    this.individuals = [];
    this.seed();
  }

  seed() {
    this.individuals = [];
    for (let i = 0; i < this.n; i++) {
      const genome = randomGenome(3);
      const sim = Sim.fromGenome(genome, this.size);
      this.individuals.push({ genome, sim, selected: false });
    }
  }

  setSize(w) {
    this.size = w;
    for (const ind of this.individuals) {
      ind.sim.setSize(w);
      ind.sim.applyGenome(ind.genome);
    }
  }

  setCount(n) {
    this.n = n;
    if (this.individuals.length < n) {
      while (this.individuals.length < n) {
        const genome = randomGenome(3);
        this.individuals.push({ genome, sim: Sim.fromGenome(genome, this.size), selected: false });
      }
    } else if (this.individuals.length > n) {
      this.individuals.length = n;
    }
  }

  // breed next generation from currently-selected winners
  breed() {
    let winners = this.individuals.filter((ind) => ind.selected);

    // gen 0: nothing selected yet → pick a random subset as the initial winners
    if (winners.length === 0 && this.generation === 0) {
      const k = 1 + Math.floor(Math.random() * this.n);
      const idxs = [...this.individuals.keys()].sort(() => Math.random() - 0.5).slice(0, k);
      idxs.forEach((i) => { this.individuals[i].selected = true; });
      winners = this.individuals.filter((ind) => ind.selected);
    }

    // nothing selected and we have prior winners → reuse the last winners,
    // augmented with a random portion of the rest of the population
    if (winners.length === 0) {
      if (this.lastWinners && this.lastWinners.length) {
        const bySig = new Map(this.lastWinners.map((g, i) => [JSON.stringify(g), i]));
        winners = this.individuals.filter((ind) => bySig.has(JSON.stringify(ind.genome)));
        // if none still match (population was rebuilt), fall back to all
        if (winners.length === 0) winners = this.individuals.slice();

        // add ~25% of the remaining (non-winner) individuals as random blood
        const rest = this.individuals.filter((ind) => !winners.includes(ind));
        const add = Math.ceil(0.25 * rest.length);
        for (let k = 0; k < add; k++) {
          const pick = rest.splice(Math.floor(Math.random() * rest.length), 1)[0];
          if (pick) winners.push(pick);
        }
        winners.forEach((w) => { w.selected = true; });
      } else {
        winners = this.individuals.slice(); // first evolve with no history → all
        winners.forEach((w) => { w.selected = true; });
      }
    }

    this.lastWinners = winners.map((w) => w.genome);

    const next = [];
    // straight copies of every winner
    for (const w of winners) {
      const sim = Sim.fromGenome(w.genome, this.size);
      next.push({ genome: w.genome, sim, selected: false });
    }
    // crossover + mutation fill the rest (children from two parents)
    while (next.length < this.n) {
      const a = winners[Math.floor(Math.random() * winners.length)];
      let childGenome;
      if (winners.length > 1 && Math.random() < this.crossoverRate) {
        const b = winners[Math.floor(Math.random() * winners.length)];
        childGenome = crossoverGenomes(a.genome, b.genome);
      } else {
        childGenome = { F: a.genome.F.map((r) => r.slice()), params: Object.assign({}, a.genome.params) };
      }
      childGenome = mutateGenome(childGenome, this.mutRate);
      const sim = Sim.fromGenome(childGenome, this.size);
      next.push({ genome: childGenome, sim, selected: false });
    }

    this.individuals = next;
    this.generation++;
    return true;
  }

  step() {
    for (let s = 0; s < this.speed; s++)
      for (const ind of this.individuals) ind.sim.step();
  }

  winners() { return this.individuals.filter((ind) => ind.selected).map((ind) => ind.genome); }
}

// ---------------------------------------------------------------------------
// Rendering + glue (p5)
// ---------------------------------------------------------------------------
let pop;
let panels = []; // { wrap, canvas, gfx }

function buildPanels() {
  const container = document.getElementById('pop');
  container.innerHTML = '';
  panels = [];
  const cols = 5; // fixed: 5 system windows per row
  container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

  // measure the real available area (viewport minus the actual bar height)
  const barH = document.getElementById('bar').getBoundingClientRect().height;
  const availW = window.innerWidth;                 // container padding
  const availH = window.innerHeight - barH - 20;         // bar + margins
  const rows = Math.ceil(pop.n / cols);
  const cell = Math.max(20, Math.floor(Math.min(availW / cols, availH / rows)));

  pop.individuals.forEach((ind, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'panel';
    const cv = document.createElement('canvas');
    cv.width = cell; cv.height = cell;
    wrap.appendChild(cv);
    container.appendChild(wrap);

    const gfx = createGraphics(pop.size, pop.size);
    gfx.pixelDensity(1); gfx.noSmooth();

    wrap.addEventListener('click', () => {
      ind.selected = !ind.selected;
      wrap.classList.toggle('selected', ind.selected);
      document.getElementById('selNum').textContent = pop.winners().length;
    });

    panels.push({ wrap, canvas: cv, gfx, cell });
  });
}

function drawIndividual(ind, panel) {
  const sim = ind.sim;
  panel.gfx.loadPixels();
  for (let i = 0; i < sim.W; i++) {
    for (let j = 0; j < sim.W; j++) {
      const p = 4 * (i * sim.W + j);
      panel.gfx.pixels[p]     = Math.min(255, sim.stores[sim._idx(i, j, 0)] * 255);
      panel.gfx.pixels[p + 1] = Math.min(255, sim.stores[sim._idx(i, j, 1)] * 255);
      panel.gfx.pixels[p + 2] = Math.min(255, sim.stores[sim._idx(i, j, 2)] * 255);
      panel.gfx.pixels[p + 3] = 255;
    }
  }
  panel.gfx.updatePixels();
  const ctx = panel.canvas.getContext('2d');
  // draw the p5 buffer onto the 2d canvas scaled to cell size
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(panel.gfx.elt, 0, 0, panel.cell, panel.cell);
}

function setup() {
  const _cnv = createCanvas(10, 10); _cnv.style('display', 'none'); // hidden master canvas
  pixelDensity(1); noSmooth();

  pop = new Population(
    parseInt(document.getElementById('popSize').value),
    parseInt(document.getElementById('gridSize').value),
    parseFloat(document.getElementById('mutRate').value),
    parseInt(document.getElementById('speed').value)
  );
  pop.crossoverRate = parseFloat(document.getElementById('crossRate').value);
  buildPanels();
  wireBar();
}

function draw() {
  pop.step();
  pop.individuals.forEach((ind, i) => drawIndividual(ind, panels[i]));
}

function windowResized() {
  buildPanels(); // recompute cell size to fit the (possibly changed) viewport
}

function wireBar() {
  document.getElementById('regenBtn').addEventListener('click', () => {
    if (pop.generation > 0 &&
        !confirm('Discard generation ' + pop.generation + ' and restart the population?')) {
      return;
    }
    pop.seed(); buildPanels();
    document.getElementById('genNum').textContent = pop.generation = 0;
    document.getElementById('selNum').textContent = 0;
  });
  document.getElementById('evolveBtn').addEventListener('click', () => {
    const ok = pop.breed();
    if (!ok) { alert('Select at least one winner first (click a panel).'); return; }
    buildPanels();
    document.getElementById('genNum').textContent = pop.generation;
    document.getElementById('selNum').textContent = 0;
  });
  document.getElementById('exportBtn').addEventListener('click', () => {
    const winners = pop.winners();
    if (winners.length === 0) { alert('No winners selected.'); return; }
    const data = { generation: pop.generation, genomes: winners };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'evolved.json';
    a.click();
  });

  // slider helper: wire a range input to a callback, updating its value label
  const slider = (id, valId, fmt, fn) => {
    const el = document.getElementById(id);
    const out = document.getElementById(valId);
    const update = () => { const v = parseFloat(el.value); fn(v); out.textContent = fmt(v); };
    el.addEventListener('input', update);
    update();
  };

  slider('popSize', 'popSizeVal', (v) => v, (v) => { pop.setCount(parseInt(v)); buildPanels(); });
  slider('gridSize', 'gridSizeVal', (v) => v, (v) => { pop.setSize(parseInt(v)); buildPanels(); });
  slider('mutRate', 'mutRateVal', (v) => v.toFixed(2), (v) => { pop.mutRate = v; });
  slider('crossRate', 'crossRateVal', (v) => v.toFixed(2), (v) => { pop.crossoverRate = v; });
  slider('speed', 'speedVal', (v) => v, (v) => { pop.speed = parseInt(v); });
}
