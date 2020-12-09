import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {Block} from './block';

export class Sidebar {
    container: Container;
    shape: Shape;
    rect: Graphics.RoundRect;
    
    sidebarBlocks: Block[]; 
    sidebarTexts: [Text, Number][];
    blocks: Block[];
    // have buttons be Blocks so that pressmove can work. on mousbuttondown add a new block to the sidebar.
    // blocks should have a sidebar state where they look different.
    constructor(stage: Stage) {
      this.container = new Container();
      this.container.x = 0;
      this.container.y = 0;

      this.blocks = [];
      this.sidebarBlocks = [];
      this.sidebarTexts = [];
  
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#fff");
      this.shape.graphics.setStrokeStyle(5);
      this.shape.graphics.beginFill("#fff");
      this.rect = new Graphics.RoundRect(0, 0, 400, 1600, 0, 0, 0, 0);
      this.shape.graphics.append(this.rect);
      this.container.addChild(this.shape);
      stage.addChild(this.container);
  
      // TODO: make sidebar blocks a different class than just Block. 
      let blockData = [
        {
          type: "separator",
          name: "layers"
        },
        {
          color: "#5B60E0",
          inputs: ['weights', 'input'],
          outputs: ['output'],
          type: "conv2d"
        },
        {
          color: "#5B60E0",
          inputs: ['weights', 'input'],
          outputs: ['output'],
          type: "dense"
        },
        {
          type: "separator",
          name: "activation functions"
        },
        {
          color: "#2B9D89",
          inputs: ['input'],
          outputs: ['output'],
          type: "relu"
        },
        {
          color: "#2B9D89",
          inputs: ['input'],
          outputs: ['output'],
          type: "sigmoid"
        },

        {
          color: "#2B9D89",
          inputs: ['input'],
          outputs: ['output'],
          type: "tanh"
        },
        {
          color: "#2B9D89",
          inputs: ['input'],
          outputs: ['output'],
          type: "softplus"
        },
        {
          type: "separator",
          name: "loss functions"
        },

        {
          color: "#D2AF53",
          inputs: ['target', 'prediction'],
          outputs: ['loss'],
          type: "cross_entropy"
        }
      ]

      let y: number = 10;
      blockData.forEach((data: any, idx: number) =>  {
        if (data.type == 'separator'){
          let text: Text = new Text(data.name, "25px Inter");
          text.x = 20;
          text.y = y + 30;
          this.sidebarTexts.push([text, text.y])
          y += 70;

          this.container.addChild(text);
        } else {
          let block: Block = new Block(stage, this.blocks, 20, y, data.color, data.inputs, data.outputs, "test", data.type, true);
          block.sidebar = true;
          this.sidebarBlocks.push(block);
          block.container.initialY = y;
          y += 60;
        }
        
      });
    }
  }
  