import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {Block} from './block';

export class Sidebar {
    container: Container;
    shape: Shape;
    rect: Graphics.RoundRect;
    
    sidebarBlocks: [Block, Number][]; 
    sidebarTexts: [Text, Number][];
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
      this.sidebarBlocks = [];
      this.sidebarTexts = [];
  
      this.shape = new Shape();
      // this.shape.graphics.beginStroke("#fff");
      // this.shape.graphics.setStrokeStyle(5);
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
              value: "[1, 1]"
            },
            "stride" : {
              type: "int | [int, int]",
              value: "[1, 1]"
            },
            "padding" : {
              type: "int | [int, int]",
              value: "[1, 1]"
            },
            "dilation" : {
              type: "int | [int, int]",
              value: "[1, 1]"
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
              value: "'zeros'"
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
          default_name: "maxpool",
          params: {
            
            "kernel_size" : {
              type: "int | [int, int]",
              value: "[2, 2]"
            },
            "stride" : {
              type: "int | [int, int]",
              value: "[2, 2]"
            },
            "padding" : {
              type: "int | [int, int]",
              value: "[0, 0]"
            },
            "dilation" : {
              type: "int | [int, int]",
              value: "[1, 1]"
            },
            
          },
          inputs: [
            
            {
              name: 'input'
            },
            {
              name: 'params',
              params: [
                
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                
              ]
            }],
          outputs: ['output'],
          type: "maxpool"
        },




        {
          color: "#5B60E0",

          count: 0,
          default_name: "dense",
          params : {
            'in_features' : {
              type: "int",
              value: "1"
            },
            'out_features' : {
              type: "int",
              value: "1"
            }
          },
          inputs: [{
            name: 'weights',
            params : [
              'in_features',
              'out_features'
            ]
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
          params: {
            "dim" : {
              type: "int",
              value: "1"
            }
          },
          count: 0,
          default_name: "softmax",
          inputs: [{
            name: "input"
          }, {
            name: "dimension",
            params: ['dim']
          }],
          outputs: ['output'],
          type: "softmax"
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
          name: "data functions"
        }, 
        {
          color: "#1480FF",

          count: 0,
          default_name: "add",
          inputs: [{
            name: 'input1'
          }, 
          {
            name: 'input2'
          }],
          outputs: ['output'],
          type: "add"
        },
        {
          color: "#1480FF",

          count: 0,
          default_name: "flatten",
          params: {
            start_dim : {
              type: "int",
              value: '1'
            },
            
            end_dim : {
              type: "int",
              value: '-1'
            }
          },
          inputs: [{
            name: 'input'
          }, 
         ],
          outputs: ['output'],
          type: "flatten"
        },
       
        
      ]

      staticObjects.push([this.container, 0, 0]);

      let y: number = 10;
      blockData.forEach((data: any, idx: number) =>  {
        if (data.type == 'separator'){
          let text: Text = new Text(data.name, "25px Inter");
          text.x = 50;
          text.y = y + 50;
          this.sidebarTexts.push([text, text.y])
          y += 90;

          this.container.addChild(text);
        }else {

          // data.params are the archetypical parameters of 
          // the block, we need to copy it so we don't edit 
          // the parameters of other blocks
          // this will happen inside the block class though
          let block: Block = new Block(stage, data.params, this.blocks, 50, y, data.color, data.inputs, data.outputs, "test", data.type, true, screen, {count: 0});
          

          staticObjects.push([block.container, 50, y]);

          block.sidebar = true;
          this.sidebarBlocks.push([block, y]);
          y += 60;
        }
        
      });
    }
  }
  