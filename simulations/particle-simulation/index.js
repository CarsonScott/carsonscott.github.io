var AREA_WIDTH = 1200;  // The width of the simulation
var AREA_HEIGHT = 700;  // The height of the simulation
var PARTICLE_COUNT = 1200;  // The number of particles
var TEMPLATE_COUNT = 2;  // The number of particle types
var STEP_SIZE = 0.1;  // The rate at which the simulation evolves
var ACCEL_RATE = 15;  // The rate at which interactive forces are applied
var REPEL_RATE = 35;  // The rate at which repulsive forces are applied
var MOUSE_RATE = 100;  // The rate at which mouse forces are applied
var BOUNDARY_RATE = 100;  // The rate at which boundary forces are applied
var SPEED_LIMIT = 15;  // The maximum speed of particles
var FRICTION_FACTOR = 0.5;  // The rate at which friction is applied
var INTERACTION_RADIUS = 120;  // The distance at which particles can interact
var MOUSE_RADIUS = 600;  // The rate at which particles are affected by mouse presses
var BACKGROUND_COLOR = [0, 0, 0, 255];  // The color of the background
var RADIUS_RANGE = [1, 5];  // The range of radius values that can be randomly assigned to a particle type
var FUNCTION_SLOPES = 5;  // The number of slopes in the randomly generated force functions
var SYMMETRIC_FORCES = false;  // Whether or not forces are equal and opposite

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Particle {
  constructor(t, x, y, r, m, c) {
      this.t = t;
      this.x = x;
      this.y = y;
      this.r = r;
      this.m = m;
      this.c = c;
      this.vx = 0;
      this.vy = 0;
  }
  
  getPos() {
    return {x:this.x, y:this.y}; 
  }
  
  applyForce(fx, fy) {
    this.vx += fx / this.m * STEP_SIZE;
    this.vy += fy / this.m * STEP_SIZE;
  }
  
  update() {
     this.x += this.vx * STEP_SIZE;
     this.y += this.vy * STEP_SIZE;
  }
  
  draw() {
    stroke(0,0);
    fill(this.c);
    circle(this.x, this.y, this.r*2);
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CustomFunction {
  points = [{x:0,y:0}, {x:1,y:0}];
  
  constructor(points) {
    for (let p of points) {
      let notAdded = true;
      for (let i=0; i<this.points.length; i++) {
        let pi = this.points[i];
        if (p.x <= pi.x) {
          if (p.x == pi.x) {
            this.points[i] = p; 
          }
          else {
            insert(this.points, i, p); 
          }
          notAdded = false;
          break;
        }
      }
      if (notAdded) {
        this.points.push(p); 
      }
    }
  }
  
  compute(x) {
    for (let i=0; i<this.points.length-1; i++) {
      if (x >= this.points[i].x && x <= this.points[i+1].x) {
        let x_rng = this.points[i+1].x - this.points[i].x;
        let y_rng = this.points[i+1].y - this.points[i].y;
        return this.points[i].y + y_rng * (x-this.points[i].x) / x_rng;
      }
    }
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function insert(X, i, x) {
  X.splice(i, 0, x);
  X.join();
  return X;
}


function distance(x, y) {
  return Math.sqrt(Math.pow(x,2) + Math.pow(y,2)); 
}


function angle(x, y) {
  return Math.atan2(y, x); 
}


function direction(x, y) {
  let a = angle(x, y);
  return {x:Math.cos(a), y:Math.sin(a)}; 
}


function distanceTo(p1, p2) {
  let dx = p2.x - p1.x;
  let dy = p2.y - p1.y;
  return distance(dx, dy);
}


function angleTo(p1, p2) {
  let dx = p2.x - p1.x;
  let dy = p2.y - p1.y;
  return angle(dx, dy);
}


function directionTo(p1, p2) {
   let a = angleTo(p1, p2);
   return {x:Math.cos(a), y:Math.sin(a)};
}


function randNum(min, max, isInt=false) {
  let tot = max - min;
  let num = min + Math.random() * tot;
  if (isInt) {
    num = Math.floor(num);
  }
  return num;
}


function randomColor(r=[0,255], g=[0,255], b=[0,255]) {
  let rng = [r, g, b];
  let c = [0, 0, 0, 255];
  for (let i=0; i<rng.length; i++) {
    c[i] = randNum(rng[i][0], rng[i][1], isInt=true); 
  }
  return c;
}


function mousePos() {
  return {x:mouseX, y:mouseY}; 
}


function pointOnCanvas(p) {
  return p.x >= 0 && p.x < AREA_WIDTH && p.y >= 0 && p.y < AREA_HEIGHT; 
}


function computeInteraction(P1, P2) {
  let p1 = P1.getPos();
  let p2 = P2.getPos();
  let dis = distanceTo(p1, p2);
  
  if (dis <= INTERACTION_RADIUS) {
    let dir = directionTo(p1, p2);
    
    let fx1;
    let fy1;
    let fx2;
    let fy2;
  
    if (dis <= P1.r + P2.r) {
      let f = (P1.r + P2.r) - dis;
      
      fx1 = -dir.x * f * Math.abs(f) * REPEL_RATE;
      fy1 = -dir.y * f * Math.abs(f) * REPEL_RATE;

      fx2 = dir.x * f * Math.abs(f) * REPEL_RATE;
      fy2 = dir.y * f * Math.abs(f) * REPEL_RATE;
    }
    
    else {  
      let F1 = interactionRules[P1.t][P2.t]
      let F2 = interactionRules[P2.t][P1.t]
      
      let x = dis / INTERACTION_RADIUS;
      
      let f1 = F1.compute(x);
      let f2 = F2.compute(x);
      
      fx1 = dir.x * f1 * ACCEL_RATE;
      fy1 = dir.y * f1 * ACCEL_RATE;
      
      fx2 = -dir.x * f2 * ACCEL_RATE;
      fy2 = -dir.y * f2 * ACCEL_RATE;
    }
    
    P1.applyForce(fx1, fy1);
    P2.applyForce(fx2, fy2);
  }
}


function applyBoundaryForce(P) {
  let p = P.getPos();
  
  let fx = 0;
  let fy = 0;
  
  if (p.x > AREA_WIDTH) {
    fx = AREA_WIDTH - p.x;
  }
  else if (p.x < 0) {
     fx = -p.x;
  }
  
  if (p.y > AREA_HEIGHT) {
    fy = AREA_HEIGHT - p.y; 
  }
  else if (p.y < 0) {
    fy = -p.y; 
  }
  
  fx = fx * Math.abs(fx) * BOUNDARY_RATE;
  fy = fy * Math.abs(fy) * BOUNDARY_RATE;
  
  P.applyForce(fx, fy);
}


function applyFrictionForce(P) {
  let fx = -P.vx * FRICTION_FACTOR;
  let fy = -P.vy * FRICTION_FACTOR;
  P.applyForce(fx, fy);
}


function applyMouseForce(P) {
  if (mouseIsPressed) {
    let p = P.getPos();
    let m = mousePos();
    
    if (pointOnCanvas(m)) {
      let dis = distanceTo(p, m);
      
      if (dis <= MOUSE_RADIUS) {
        let dir = directionTo(p, m);
        let f = (1 - dis / MOUSE_RADIUS) * (Number(mouseButton == LEFT) - Number(mouseButton == RIGHT));
        
        let fx = dir.x * f * MOUSE_RATE;
        let fy = dir.y * f * MOUSE_RATE;
        
        P.applyForce(fx, fy);
      }
    }
  }
}


function applySpeedLimit(P) {
  if (SPEED_LIMIT != null) {
    let s = Math.sqrt(Math.pow(P.vx,2) + Math.pow(P.vy,2));
    if (s > SPEED_LIMIT) {
      let dir = direction(P.vx, P.vy);
      P.vx = dir.x * SPEED_LIMIT;
      P.vy = dir.y * SPEED_LIMIT;
    }
  }
}


function create2dArray(h, w) {
  let array = [];
  for (let i=0; i<h; i++) {
    array.push(new Array(w)); 
  }
  return array;
}


function createRandomFunction(n) {
  let points = [];
  for (let i=0; i<n; i++) {
    let p = {x:randNum(0,1), y:randNum(-1,1)};
    points.push(p);
  }
  return new CustomFunction(points);
}


function createParticleFromTemplate(t, x, y) {
   return new Particle(t.t, x, y, t.r, t.m, t.c);
}


function generateParticleTemplates() {
  let particleTemplates = [];
  for (let i=0; i<TEMPLATE_COUNT; i++) {
    let r = randNum(RADIUS_RANGE[0], RADIUS_RANGE[1]);
    let m = r * 0.75;
    let c = randomColor();
    let t = {t:i, r:r, m:m, c:c};
    particleTemplates.push(t);
  }
  return particleTemplates;
}


function generateInteractionRules() {
  interactionRules = create2dArray(TEMPLATE_COUNT, TEMPLATE_COUNT);
  for (let i=0; i<TEMPLATE_COUNT; i++) {
    if (SYMMETRIC_FORCES) {
      for (let j=i; j<TEMPLATE_COUNT; j++) {
        let F = createRandomFunction(FUNCTION_SLOPES);
        interactionRules[i][j] = F;
        interactionRules[j][i] = F;
      }
    }
    else {
      for (let j=0; j<TEMPLATE_COUNT; j++) {
        let F = createRandomFunction(FUNCTION_SLOPES);
        interactionRules[i][j] = F;
      }
    }
  }
  return interactionRules;
}


function generateParticles() {
  particles = [];
  
  for (let i=0; i<PARTICLE_COUNT; i++) {
    let j = randNum(0, TEMPLATE_COUNT, isInt=true);
    let t = particleTemplates[j];
    let x = randNum(0, AREA_WIDTH);
    let y = randNum(0, AREA_HEIGHT);
    let p = createParticleFromTemplate(t, x, y);
    particles.push(p);
  }
  
  return particles;
}


function updateParticles() {
  for (let i=0; i<particles.length; i++) {
    let Pi = particles[i];
    for (let j=i; j<particles.length; j++) {
      if (i != j) {
        let Pj = particles[j];
        computeInteraction(Pi, Pj);
      }
    }
    
    applyMouseForce(Pi);
    applyBoundaryForce(Pi);
    applySpeedLimit(Pi);
    applyFrictionForce(Pi);
    Pi.update();
    Pi.draw();
  }
}


function generate() {
  particleTemplates = generateParticleTemplates();
  interactionRules = generateInteractionRules();
  particles = generateParticles();
}


function randomize() {
  for (let p of particles) {
    p.x = randNum(0, AREA_WIDTH);
    p.y = randNum(0, AREA_HEIGHT);
    p.vx = 0;
    p.vy = 0;
  }
}


function toggleMenu() {
  let toggle = document.getElementById("toggle-menu");
  let menu = document.getElementById("menu-container");
  let style = getComputedStyle(menu);
  let visibility = style.getPropertyValue("visibility");

  if (visibility == "visible") {
    menu.style.setProperty("visibility", "hidden");
    toggle.value = "Show Menu";
  }
  else if (visibility == "hidden") {
    menu.style.setProperty("visibility", "visible");
    toggle.value = "Hide Menu";
  }
}


function updateSlider(slider) {
  let sliderValue = slider.nextElementSibling;
  sliderValue.innerHTML = slider.value;
}


function initializeSliders() {
  let containers = document.getElementsByClassName("slider-container");
  for (let container of containers) {
    let slider = container.getElementsByClassName("slider")[0];
    updateSlider(slider);
  }
}


function initializeMenu() {
  document.oncontextmenu = function() {return false};
  document.getElementById("generate").onclick = generate;
  document.getElementById("randomize").onclick = randomize;
  document.getElementById("toggle-menu").onclick = toggleMenu;
  document.getElementById("symmetric-forces").checked = SYMMETRIC_FORCES;
  document.getElementById("particle-count").value = PARTICLE_COUNT;
  document.getElementById("template-count").value = TEMPLATE_COUNT;
  initializeSliders();
  
  document.getElementById("particle-count").oninput = function() {
    updateSlider(this);
    PARTICLE_COUNT = this.value;
  }
  document.getElementById("template-count").oninput = function() {
    updateSlider(this);
    TEMPLATE_COUNT = this.value;
  }
  document.getElementById("symmetric-forces").oninput = function() {
    SYMMETRIC_FORCES = this.checked;
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

var particleTemplates;
var interactionRules;
var particles;

function setup() {
  createCanvas(AREA_WIDTH, AREA_HEIGHT);
  initializeMenu();
  generate();
}

function draw() {
  background(BACKGROUND_COLOR);
  updateParticles();  
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~