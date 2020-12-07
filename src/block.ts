import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import * as util from './util';

import {EditorElement} from './editorElement';
import {Inputs, Input} from './inputs';
import {Output, Outputs} from './outputs';
import {Sidebar} from './sidebar'

export class Block {
    container: Container; 
    blockBody: BlockBody;
    inputs: Inputs;
    outputs: Outputs;
    sidebar: Sidebar;  // the sidebar, if any, that this block belongs to. display the compact sidebar version of this block
  
    blocks: Block[] = [];   // the list of all other blocks, since this is pass by reference we chillin on the memory usage B)

    clickOffset: [number, number];
    //outputs: Output;

    children: [Input, Output][];
  
    constructor(stage: Stage, blocks: Block[], x: number, y: number, color: String, inputs: String[], outputs: String[], name: String, type: String, sidebar: Sidebar) {
      console.log(blocks);
      this.sidebar = sidebar;
  
      this.container = new Container();
      this.container.x = x;
      this.container.y = y;
     
      this.blocks = blocks;
      
      
      this.children = [];
      
      this.blockBody = new BlockBody({
          color: color,
          type: type,
          name: name,
          isSidebar: sidebar != null
       });
      this.inputs = new Inputs({
        inputs: inputs,
        //blocks: this.blocks
      });
      this.outputs = new Outputs({
        inputs: outputs,
        block: this,
        checkConnection: this.checkConnection
      });
  
      this.container.addChild(this.blockBody.container);

      if (this.sidebar == null) {
        this.container.addChild(this.inputs.container);
        this.container.addChild(this.outputs.container);
      }
      
      stage.addChild(this.container);
  
      // attach listeners
      this.blockBody.container.addEventListener('mousedown', (event) => {
        console.log(event);
        if (this.sidebar != null) {
          if (this == this.sidebar.newConv) {
            let newBlock: Block = new Block(stage, this.blocks,
              this.sidebar.container.x + this.sidebar.newConvXOffset, this.sidebar.container.y + this.sidebar.newConvYOffset,
             "#5B60E0", ["weights", "input", ], ['output'], "conv_1", "convolution-", this.sidebar);
            
            this.blocks.push(newBlock)
            this.sidebar.newConv = newBlock;
          }
          else if (this == this.sidebar.newAttentive) {
            let newBlock: Block = new Block(stage, this.blocks,
                this.sidebar.container.x + this.sidebar.newAttentiveXOffset, this.sidebar.container.y + this.sidebar.newAttentiveYOffset,
                "#F97979", ["canvas", "intensity", "gx", "gy", "stride", "variance"], ['output2'], "write", "attentive_write-", this.sidebar);
            
            this.blocks.push(newBlock);
            this.sidebar.newAttentive = newBlock;
          } 
          this.sidebar = null
          this.container.addChild(this.inputs.container);
          this.container.addChild(this.outputs.container);
          this.blockBody.isSidebar = false;
          this.blockBody.completeText();
          this.update();
        }
        
        let localPos = stage.globalToLocal(event.stageX, event.stageY);
        this.clickOffset = [this.container.x - localPos.x,
                                  this.container.y - localPos.y];
        
      })
  
      this.blockBody.container.addEventListener('pressmove', (event) => {
        let localPos = stage.globalToLocal(event.stageX, event.stageY);
        this.container.x = localPos.x + this.clickOffset[0];
        this.container.y = localPos.y + this.clickOffset[1];

        this.outputs.updateConnections();
        this.inputs.updateConnections();
      })
  
      this.update();
    }
  
    checkConnection = (event: any, output: Output) => {
      let x: number = event.stageX;
      let y: number = event.stageY; 
      //console.log(this.blocks);
      this.blocks.forEach((block: Block, index: number) => {
        if (block != this) {
          block.inputs.inputs.forEach((inp: Input, index: number) => {
            let local: Point = inp.container.globalToLocal(x, y);
            //console.log(local + " " + inp.props.name);
            if (inp.container.hitTest(local.x, local.y)){
              console.log(inp.props.name);
              // make the connection
              //output.addConnection(inp);
              this.outputs.addParent([output, inp]);
              block.inputs.addChild([output, inp]);
            }
          })
        }
      });
  
    }
  
    update = () => {
      // find the bounds of BlockBody
      let maxWidth: number = 0;
  
      if (!this.blockBody.isSidebar) {
        maxWidth = util.updateMax(maxWidth, this.inputs.update());
        maxWidth = util.updateMax(maxWidth, this.outputs.update());
        maxWidth = util.updateMax(maxWidth, this.blockBody.container.getBounds().width);
        this.blockBody.update(maxWidth + 2 * constants.TEXT_PADDING_LR);
      }
      // after the maxWidth has been computed, position everything
      
      // position inputs
      this.inputs.container.x = (maxWidth - this.inputs.container.getBounds().width) / 2;
      this.inputs.container.y = constants.TEXT_PADDING_UD + constants.IO_PADDING_UD + constants.IO_TEXT_PADDING_UD * 5;
  
      // position outputs
      this.outputs.container.x = (maxWidth - this.outputs.container.getBounds().width) / 2;
      this.outputs.container.y = -constants.TEXT_PADDING_UD - constants.IO_PADDING_UD - constants.IO_TEXT_PADDING_UD * 3;

     
    }

    createConnections = () => {

    }
  }

export class BlockBody extends EditorElement{
    shape: Shape;
    rect: Graphics.RoundRect;
    container: Container;
    isSidebar: Boolean;
  
    blockType: Text;
    blockName: Text; 
    constructor(props) {
      super(props);
  
      this.isSidebar = props.isSidebar;
  
      this.container = new Container();
      this.shape = new Shape();
  
      this.shape.graphics.beginStroke(props.color);
      this.shape.graphics.setStrokeStyle(5);
      this.shape.graphics.beginFill("#fff");
  
      this.container.addChild(this.shape);
  
      // x, y, w, h, corners x 4
      // default values here are for sidebar display
      this.rect = new Graphics.RoundRect(-constants.TEXT_PADDING_LR, -10,
        180, 60,
        30, 30, 30, 30)
        
      // create text
      if (props.isSidebar) {
        this.blockName = new Text("new", "40px Inter", props.color);
      } else {
        this.completeText();
      }
  
      this.container.addChild(this.blockName);
  
      
  
      this.shape.graphics.append(this.rect);
    }
  
    completeText = () => {
      console.log("COMPLETE TEXT:" + this.props.name + this.props.color)
      this.container.removeChild(this.blockName)
      this.blockName = new Text(this.props.name, "40px Inter", this.props.color);
      this.blockType = new Text(this.props.type + " :: ", "40px Inter", "#000");
      this.blockName.x = this.blockType.getBounds().width;
      this.container.addChild(this.blockType);
      this.container.addChild(this.blockName);
      this.rect.radiusTL = 100;
      this.rect.radiusTR = 100;
      this.rect.radiusBR = 100;
      this.rect.radiusBL = 100;
      this.rect.y = -constants.TEXT_PADDING_UD - 3;
    }
  
    update = (width: number) => {
      console.log(width);
  
      let bounds: Rectangle = this.container.getBounds();
  
      this.rect.w = width;
      this.rect.h = bounds.height + constants.TEXT_PADDING_UD * 2;
  
      let totalTextWidth: number = this.blockType.getBounds().width + this.blockName.getBounds().width;
      this.blockType.x = (width - 2 * constants.TEXT_PADDING_LR - totalTextWidth) / 2;
      this.blockName.x = this.blockType.getBounds().width + (width - 2 * constants.TEXT_PADDING_LR - totalTextWidth) / 2;
      
    }
  }