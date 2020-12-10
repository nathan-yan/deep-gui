import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {Block} from './block';

export class Sidebar {
    container: Container;
    shape: Shape;
    rect: Graphics.RoundRect;
    
    sidebarBlocks: Block[]; 
    blocks: Block[];
    
    screen: Container;
    // have buttons be Blocks so that pressmove can work. on mousbuttondown add a new block to the sidebar.
    // blocks should have a sidebar state where they look different.
    constructor(stage: Stage, screen: Container, staticObjects: [Shape, number, number][]) {
      this.container = new Container();
      this.container.x = 0;
      this.container.y = 0;

      this.screen = screen;

      this.blocks = [];
  
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#ddd");
      this.shape.graphics.setStrokeStyle(5);
      this.shape.graphics.beginFill("#fff");
      this.rect = new Graphics.RoundRect(0, 0, 400, 1600, 0, 0, 0, 0);
      this.shape.graphics.append(this.rect);
      this.container.addChild(this.shape);
      stage.addChild(this.container);
  
      // TODO: make sidebar blocks a different class than just Block. 
      // TODO: move this block data mess to a different file, or maybe a config file
      let blockData = [
        {
          type: "separator",
          name: "layers"
        },
        {
          color: "#5B60E0",
          count: 0,
          default_name: "conv",
          params: {
            "in_channels" : {
              type: "int",
              value: 1
            }, 
            "out_channels" : {
              type: "int",
              value: 1
            } ,
            "kernel_size" : {
              type: "int | [int, int]",
              value: [1, 1]
            },
            "stride" : {
              type: "int | [int, int]",
              value: [1, 1]
            },
            "padding" : {
              type: "int | [int, int]",
              value: [0, 0]
            },
            "dilation" : {
              type: "int | [int, int]",
              value: [1, 1]
            },
            "groups" : {
              type: "int",
              value: 1
            },

            "bias" : {
              type: "bool",
              value: "True"
            },
            "padding_mode" : {
              type: "zeros | reflect | replicate | circular",
              value: "zeros"
            } 
          },
          inputs: [
            {
              name: 'weights',
              params: [
                'in_channels',
                'out_channels',
                'kernel_size'
              ]
            }, 
            {
              name: 'input'
            },
            {
              name: 'params',
              params: [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "padding_mode",
                
              ]
            }],
          outputs: ['output'],
          type: "conv2d"
        },
        {
          color: "#5B60E0",

          count: 0,
          default_name: "dense",
          inputs: [{
            name: 'weights'
          }, 
          {
            name: 'input'
          }],
          outputs: ['output'],
          type: "dense"
        },
        {
          type: "separator",
          name: "activation functions"
        },
        {
          color: "#2B9D89",

          count: 0,
          default_name: "relu",
          inputs: [{
            name: "input"
          }],
          outputs: ['output'],
          type: "relu"
        },
        {
          color: "#2B9D89",

          count: 0,
          default_name: "sigmoid",
          inputs: [{
            name: "input"
          }],
          outputs: ['output'],
          type: "sigmoid"
        },

        {
          color: "#2B9D89",

          count: 0,
          default_name: "tanh",
          inputs: [{
            name: "input"
          }],
          outputs: ['output'],
          type: "tanh"
        },
        {
          color: "#2B9D89",

          count: 0,
          default_name: "softplus",
          inputs: [{
            name: "input"
          }],
          outputs: ['output'],
          type: "softplus"
        },
        {
          type: "separator",
          name: "loss functions"
        },

        {
          color: "#D2AF53",

          count: 0,
          default_name: "xent",
          inputs: [{
            name: 'prediction'
          }, 
          {
            name: 'target'
          }],
          outputs: ['loss'],
          type: "cross_entropy"
        },
        {
          type: "separator",
          name: "data functions"
        }, 
        {
          color: "#1480FF",
          params: {
            "file path" : {
              type: ".grad",
              value: ""
            },
            "batch_size" : {
              type: "int",
              value: 1
            },
            "shuffle" : {
              type: "bool",
              value: "True"
            }
          },
          count: 0,
          default_name: "input",
          inputs: [{
            name: 'path',
            params: [
              "file path",
              "shuffle"
            ]
          } 
          ],
          outputs: ['data', 'target'],
          type: "input_data"
        }
      ]

      staticObjects.push([this.container, 0, 0]);

      let y: number = 10;
      blockData.forEach((data: any, idx: number) =>  {
        if (data.type == 'separator'){
          let text: Text = new Text(data.name, "25px Inter");
          text.x = 30;
          text.y = y + 30;
          y += 70;

          this.container.addChild(text);
        }else {

          // data.params are the archetypical parameters of 
          // the block, we need to copy it so we don't edit 
          // the parameters of other blocks
          // this will happen inside the block class though
          let block: Block = new Block(stage, data.params, this.blocks, 30, y, data.color, data.inputs, data.outputs, "test", data.type, true, screen);
          

          staticObjects.push([block.container, 30, y]);

          block.sidebar = true;
          y += 60;
        }
        
      });
    }
  }
  