import { Stage, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as util from "./util";

const TEXT_PADDING_LR = 60;
const TEXT_PADDING_UD = 30;
const IO_PADDING_LR = 10;
const IO_PADDING_UD = 20;
const IO_TEXT_PADDING_LR = 15;
const IO_TEXT_PADDING_UD = 10;

class BlockInput {
  container: Container;            // the container that actually holds the input
  inputRectShape: Shape;
  inputRect: Graphics.RoundRect;
  inputName: Text;

  constructor(name: String) {
    this.container = new Container();
  }
}

class Block {
  // define all drawables
  container: Container;             // the container that actually holds the entire block
  blockRectShape: Shape;            // the shape representing the block's body
  blockRect: Graphics.RoundRect;    // the rounded rectangle which actually represents the block's body
  blockTypeText: Text;              // the text object representing block type
  blockNameText: Text;              // the text object representing block name

  inputRectShapes: Shape[] = [];
  inputRects: Graphics.RoundRect[] = [];
  inputNameTexts: Text[] = [];

  state: BlockState;
  color: String;

  constructor(stage: Stage, x: number, y: number, color: String, inputs: String[]) {
    // create the block child and add it to the stage
    this.container = new Container();
    this.state = new BlockState();

    this.container.x = x;
    this.container.y = y;

    this.blockRectShape = new Shape();
    this.blockRectShape.graphics.beginStroke(color);
    this.blockRectShape.graphics.setStrokeStyle(5);
    this.blockRectShape.graphics.beginFill("#fff");

    this.container.addChild(this.blockRectShape);

    // create text
    this.blockTypeText = new Text("attentive_write :: ", "40px Inter", "#000000");
    this.blockNameText = new Text("write", "40px Inter", color);

    this.container.addChild(this.blockTypeText);
    this.container.addChild(this.blockNameText);

    this.blockRect = new Graphics.RoundRect(0, 0, 
                                            100, 70, 
                                            100, 100, 100, 100)
    this.blockRectShape.graphics.append(this.blockRect);

    // add the inputs
    inputs.forEach((value: String, index: Number) => {
      // create a block shape
      let inputRectShape: Shape = new Shape();
      inputRectShape.graphics.beginStroke("#000");
      inputRectShape.graphics.setStrokeStyle(2);
      inputRectShape.graphics.beginFill("#fff");

      let inputNameText = new Text(value, "20px Inter", "#000000"); 

      let inputRect: Graphics.RoundRect = new Graphics.RoundRect(0, 0, 
        120, 40, 
        100, 100, 100, 100)
      
      inputRectShape.graphics.append(inputRect);

      this.inputNameTexts.push(inputNameText);
      this.inputRectShapes.push(inputRectShape);
      this.inputRects.push(inputRect);
      this.container.addChild(inputRectShape);
      this.container.addChild(inputNameText);
    });

    // attach listeners
    this.container.addEventListener('mousedown', (event) => {
      console.log(event);
      this.state.clickedOn = true;
      this.state.clickOffset = [this.container.x - event.stageX,
                                this.container.y - event.stageY];
    })

    this.container.addEventListener('pressmove', (event) => {
      this.container.x = event.stageX + this.state.clickOffset[0];
      this.container.y = event.stageY + this.state.clickOffset[1];;
    })

    stage.addChild(this.container);

    this.handleBounds();
  }

  handleBounds = () => {
    let maxWidth: number = 0;

    // get blockTypeText and blockNameText rects
    let blockTypeRect: Rectangle = this.blockTypeText.getBounds();
    let blockNameRect: Rectangle = this.blockNameText.getBounds();

    let textWidth: number = blockTypeRect.width + blockNameRect.width + 
                            2 * TEXT_PADDING_LR;

    maxWidth = util.updateMax(maxWidth, textWidth);

    // TODO: get input and output rects
    let inputWidth: number = 0;
    this.inputRects.forEach((rect: Graphics.RoundedRect, idx) => {
      console.log(rect.w);
      inputWidth += rect.w + IO_PADDING_LR;
    })

    maxWidth = util.updateMax(maxWidth, inputWidth);
    
    this.blockRect.w = maxWidth;
    this.blockRect.h = Math.max(blockTypeRect.height, blockNameRect.height)+
                      2 * TEXT_PADDING_UD;

    // position type and name
    let offset: number = (maxWidth - textWidth) / 2 
    this.blockTypeText.x = offset + TEXT_PADDING_LR;
    this.blockNameText.x = offset + blockTypeRect.width + TEXT_PADDING_LR;
    this.blockTypeText.y = TEXT_PADDING_UD + 3;
    this.blockNameText.y = TEXT_PADDING_UD + 3;

    // position input
    offset = (maxWidth - inputWidth) / 2;
    let accWidth: number = 0;
    this.inputRects.forEach((rect: Graphics.RoundedRect, idx) => {
      rect.x = offset + accWidth;
      rect.y = this.blockRect.h + IO_PADDING_UD;
      rect.w = this.inputNameTexts[idx].getBounds().width + IO_TEXT_PADDING_LR * 2;
      rect.h = this.inputNameTexts[idx].getBounds().height + IO_TEXT_PADDING_UD * 2;

      this.inputNameTexts[idx].x = offset + accWidth + IO_TEXT_PADDING_LR;
      this.inputNameTexts[idx].y = this.blockRect.h + IO_PADDING_UD + IO_TEXT_PADDING_UD;

      accWidth += rect.w + IO_PADDING_LR + IO_TEXT_PADDING_LR * 2;
    })
  }
}

class BlockState {
  clickedOn: boolean = false;
  clickOffset: [number, number]; 
}

class State {
  blocks: [Block];
}

function tickGenerator(stage: Stage, state: State) {
  return function tick(event: Event) {
    stage.update();
    
  }
}

window.addEventListener("load", () => {
  //get the canvas, canvas context, and dpi
  let canvas = <HTMLCanvasElement> document.getElementById('myCanvas'),
  ctx = canvas.getContext('2d'),
  dpi = window.devicePixelRatio * 2;

  canvas.width = 1000;
  canvas.height = 800;
  canvas.style.width = "500px";
  canvas.style.height = "400px";
  ctx.scale(2, 2);

  //Create a stage by getting a reference to the canvas
  let stage = new Stage("myCanvas");
  let block: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]);

  let state = new State();
  state.blocks = [block];
  
  Ticker.framerate = 60;
  Ticker.addEventListener('tick', tickGenerator(stage, state));  
})
