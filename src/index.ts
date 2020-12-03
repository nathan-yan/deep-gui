import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as util from "./util";
import * as constants from "./constants";

import {EditorElement} from "./editorElement";
import {Block, BlockBody} from "./block";

import {Sidebar} from './sidebar';

// store all non-moving elements
let staticObjects: any[] = new Array();

function tickGenerator(stage: Stage) {
  return function tick(event: Event) {
    stage.update();
    
  }
}

// scale function
function scale(stage, zoom, zoomPt, staticObjects) {
  // set zoom bounds
  if (zoom > 1 && stage.scale < 5 || zoom < 1 && stage.scale > 0.2) {
    let localPos = stage.globalToLocal(zoomPt[0], zoomPt[1]);
    stage.regX = localPos.x;
    stage.regY = localPos.y;
    stage.x = zoomPt[0];
    stage.y = zoomPt[1];

    stage.scale *= zoom;

    staticObjects.forEach(object => {
      object[0].graphics.command.w /= zoom;
      object[0].graphics.command.h /= zoom;

      let objPos = stage.globalToLocal(object[1], object[2]);
      object[0].graphics.command.x = objPos.x;
      object[0].graphics.command.y = objPos.y;
      console.log(stage.scale);
    })

    stage.update();
  }
}

// pan function
function pan(stage, screen, staticObjects) {
  screen.addEventListener("mousedown", (event1) => {
    let initPos = [stage.x, stage.y];

    screen.addEventListener('pressmove', (event2) => {
      stage.x = initPos[0] + event2.stageX - event1.stageX;
      stage.y = initPos[1] + event2.stageY - event1.stageY;
      
      staticObjects.forEach(object => {
        let pos = stage.globalToLocal(object[1], object[2]);
        object[0].graphics.command.x = pos.x;
        object[0].graphics.command.y = pos.y;
      })      
    })
  })
}

function zoomButtons(stage, canvas, zoomIntensity) {
  // zoom buttons (might be better to replace with html buttons)
  let zoomIn = new Shape();
  zoomIn.graphics.beginFill("white").drawRect(25, 25, 50, 50);
  staticObjects.push([zoomIn, 25, 25]);
  stage.addChild(zoomIn);

  zoomIn.addEventListener("click", (event) => {
    scale(stage, zoomIntensity, [canvas.width/2, canvas.height/2], staticObjects);
  })

  let zoomOut = new Shape();
  zoomOut.graphics.beginFill("white").drawRect(25, 100, 50, 50);
  staticObjects.push([zoomOut, 25, 100]);
  stage.addChild(zoomOut);

  zoomOut.addEventListener("click", (event) => {
    scale(stage, 1/zoomIntensity, [canvas.width/2, canvas.height/2], staticObjects);
  })
}

window.addEventListener("load", () => {
  //get the canvas, canvas context, and dpi
  
  let canvas = <HTMLCanvasElement> document.getElementById('myCanvas'),
  ctx = canvas.getContext('2d'),
  dpi = window.devicePixelRatio * 2;

  canvas.width = 2000;
  canvas.height = 1600;
  canvas.style.width = "1000px";
  canvas.style.height = "800px";
  ctx.scale(2, 2);

  //Create a stage by getting a reference to the canvas
  let stage = new Stage("myCanvas");
  stage.enableMouseOver(10);
  
  // set up a customizable background screen
  let screen = new Shape();
  screen.graphics.beginLinearGradientFill(["#CC91FF" ,"#91A9FF"], [0, 1], -2*canvas.width, -2*canvas.height, 2*canvas.width, 2*canvas.height).drawRect(0, 0, canvas.width, canvas.height);
  staticObjects.push([screen, 0, 0]);
  stage.addChild(screen);

  let sidebar = new Sidebar(stage)

  // change how much stage zooms each step
  let zoomIntensity = 1.2;

  // mouse wheel zoom
  canvas.addEventListener("wheel", (event) => {
    let zoom = event.deltaY < 0 ? 1/zoomIntensity : zoomIntensity;
    scale(stage, zoom, [stage.mouseX, stage.mouseY], staticObjects);
  });

  // click and drag pan
  pan(stage, screen, staticObjects);

  let block: Block = new Block(stage, 200, 200, "#5B60E0", ["weights", "input", ], ['output'], "conv_1", "convolution", null);
  let block2: Block = new Block(stage, 200, 500, "#F97979", ["canvas", "intensity", "gx", "gy", "stride", "variance"], ['output2'], "write", "attentive_write", null);
  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]); 

  // zoom buttons
  zoomButtons(stage, canvas, zoomIntensity);
  
  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]);

  Ticker.framerate = 60;
  Ticker.addEventListener('tick', tickGenerator(stage));  
})
