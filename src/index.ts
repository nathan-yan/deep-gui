import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as util from "./util";
import * as constants from "./constants";
import * as monaco from 'monaco-editor';

import {EditorElement} from "./editorElement";
import {Block, BlockBody} from "./block";

import {Sidebar} from './sidebar';
import { Output } from "./outputs";
import { Input } from "./inputs";

// store all non-moving elements
let staticObjects: [Shape, number, number][] = new Array();

// scale function
function scale(stage, zoom, zoomPt, staticObjects) {
  // set zoom bounds
  
  let localPos = stage.globalToLocal(zoomPt[0], zoomPt[1]);
  stage.regX = localPos.x;
  stage.regY = localPos.y;
  stage.x = zoomPt[0];
  stage.y = zoomPt[1];

  stage.scale = zoom;

    /*
    staticObjects.forEach(object => {
      object[0].graphics.command.w /= zoom;
      object[0].graphics.command.h /= zoom;

      let objPos = stage.globalToLocal(object[1], object[2]);
      object[0].graphics.command.x = objPos.x;
      object[0].graphics.command.y = objPos.y;
      console.log(stage.scale);
    })*/

    //stage.update();
}

// pan function
function pan(stage, screen, staticObjects) {
  screen.addEventListener("mousedown", (event1) => {
    let initPos = [stage.x, stage.y];

    screen.addEventListener('pressmove', (event2) => {
      stage.x = initPos[0] + event2.stageX - event1.stageX;
      stage.y = initPos[1] + event2.stageY - event1.stageY;
      
      /*
      staticObjects.forEach(object => {
        let pos = stage.globalToLocal(object[1], object[2]);
        object[0].x = pos.x;
        object[0].y = pos.y;
      })     */ 
    })
  })
}

function resetSidebarPos(stage, sidebar, zoom=1){
  let sidebarPos = stage.globalToLocal(0, 0);
  sidebar.shape.graphics.command.x = sidebarPos.x;
  sidebar.shape.graphics.command.y = sidebarPos.y;

  sidebar.sidebarBlocks.forEach(sidebarBlock => {
    sidebarBlock[0].blockBody.container.scale /= zoom;
    
    let blockPos = stage.globalToLocal(20, sidebarBlock[1]);
    sidebarBlock[0].container.x = blockPos.x;
    sidebarBlock[0].container.y = blockPos.y;
  })

  sidebar.sidebarTexts.forEach(sidebarText => {
    sidebarText[0].scale /= zoom;
    
    let textPos = stage.globalToLocal(20, sidebarText[1]);
    sidebarText[0].x = textPos.x;
    sidebarText[0].y = textPos.y;
  })
}

function zoomButtons(stage, canvas, zoomIntensity, sidebar) {
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

class Editor extends Stage {
  constructor(id) {
    super(id);
    console.log("this is the editor!");

    super.addEventListener("mousedown", (event) => {
      console.log("editor");
      console.log(event)
    });
  }
}

function tickGenerator(stage: Stage) {
  return function tick(event: Event) {
    stage.update();
    
  }
}

window.addEventListener("load", () => {
  //get the canvas, canvas context, and dpi
  // set up the code editor
  /*
  monaco.editor.create(document.getElementById('codeEditor'), {
    value: [
      'function x() {',
      '\tconsole.log("Hello world!");',
      '}'
    ].join('\n'),
    language: 'javascript'
  });*/

  let canvas = <HTMLCanvasElement> document.getElementById('myCanvas');
  let blocks: Block[] = [];
  let sidebarBlocks: Block[] = [];

  let displayContainer = new Container();
  
  displayContainer.x = 300;
  displayContainer.y = 0;

  let ctx = canvas.getContext('2d'),
  dpi = window.devicePixelRatio * 2;

  canvas.width = 2000;
  canvas.height = 1400;
  canvas.style.width = "1000px";
  canvas.style.height = "700px";
  ctx.scale(2, 2);

  //Create a stage by getting a reference to the canvas
  let stage = <Stage> new Editor("myCanvas");
  stage.enableMouseOver(10);
  
  // set up a customizable background screen
  let screen = new Shape();
  screen.graphics.beginLinearGradientFill(["#fafafa" ,"#fafafa"], [0, 1], -2*canvas.width, -2*canvas.height, 2*canvas.width, 2*canvas.height).drawRect(0, 0, canvas.width, canvas.height);
  staticObjects.push([screen, 0, 0]);
  stage.addChild(screen);
  stage.addChild(displayContainer);

  let sidebar = new Sidebar(stage, displayContainer, staticObjects);

  // change how much stage zooms each step
  let zoomIntensity = 1.2;
  let zoom = 1;

  // mouse wheel zoom
  canvas.addEventListener("wheel", (event) => {
    //console.log(event.deltaY + " " + zoom);
    //let zoom = event.deltaY < 0 ? 1/zoomIntensity : zoomIntensity;
    zoom += event.deltaY / 100;
    if (zoom < 0.1) {
      zoom = 0.1
    }else if (zoom > 3) {
      zoom = 3
    }

    scale(displayContainer, zoom, [stage.mouseX, stage.mouseY], staticObjects);
  });

  document.getElementById("compile-button").onclick = () => {
    // get all the blocks

    // each block is in the format:
    /*
      block_name: {
       
        inputs: [   // where this block's inputs come from
          past_block_name.output_name, ...
        ],
        attributes : [
          attr1, ...
        ]
      }

      // if an input has params, it is not considered a real input and gets placed into `attributes`
      // however if the input has a connection, it IS considered a real input and gets placed into `inputs`
    */
    let jsonNetwork = {};
    console.log(sidebar.blocks);
    sidebar.blocks.forEach((block: Block, idx: number) => {
      let blockData = {
        inputs: {},
        attributes: [],
        type: block.blockBody.blockType.text //oof 
      };

      block.inputs.children.forEach((pair: [Output, Input], idx: number) => {
        blockData.inputs[pair[1].props.name] = pair[0].props.block.blockBody.name + "." + pair[0].props.name;
      })

      let seenParams = new Set();
      block.inputs.inputs.forEach((inp: Input, idx: number) => {
        if (inp.props.availableParams) {
          inp.props.availableParams.forEach((param, idx) => {
            if (!seenParams.has(param)){
              blockData.attributes.push(param + "=" + inp.props.params[param].value);
              seenParams.add(param);
            }
          })
        }
      })
      
      jsonNetwork[block.blockBody.name] = blockData;
    });

    console.log(JSON.stringify(jsonNetwork));
  }

  // click and drag pan
  pan(displayContainer, screen, staticObjects);

  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]); 

  // zoom buttons
  //zoomButtons(stage, canvas, zoomIntensity);
  
  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]);

  Ticker.framerate = 60;
  Ticker.addEventListener('tick', tickGenerator(stage));  
})
