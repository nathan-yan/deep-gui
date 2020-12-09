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
    sidebar: boolean;  // the sidebar, if any, that this block belongs to. display the compact sidebar version of this block
  
    blocks: Block[] = [];   // the list of all other blocks, since this is pass by reference we chillin on the memory usage B)

    clickOffset: [number, number];
    //outputs: Output;

    children: [Input, Output][];

    focusedNewBlock: Block;
  
    constructor(stage: Stage, blocks: Block[], x: number, y: number, color: String, inputs: String[], outputs: String[], name: String, type: String, sidebar: boolean) {
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
          isSidebar: sidebar == true
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

      if (!this.sidebar) {
        this.container.addChild(this.inputs.container);
        this.container.addChild(this.outputs.container);
      }

      stage.addChild(this.container);
  
      // attach listeners
      this.blockBody.container.addEventListener('mousedown', (event) => {
        let localPos = stage.globalToLocal(event.stageX, event.stageY);
        this.clickOffset = [this.container.x - localPos.x,
                                  this.container.y - localPos.y];

        if (this.sidebar) {
          
            let newBlock: Block = new Block(stage, this.blocks,
                localPos.x + this.clickOffset[0], localPos.y + this.clickOffset[1],
                color,  inputs, outputs, name, type, false);
            
            this.blocks.push(newBlock);
            this.focusedNewBlock = newBlock;
          } 
        
      })
  
      this.blockBody.container.addEventListener('pressmove', (event) => {
        let localPos: Point = stage.globalToLocal(event.stageX, event.stageY);

        if (!this.sidebar){
          this.container.x = localPos.x + this.clickOffset[0];
          this.container.y = localPos.y + this.clickOffset[1];

          this.outputs.updateConnections();
          this.inputs.updateConnections();
        }else {
          this.focusedNewBlock.container.x = localPos.x + this.clickOffset[0];
          this.focusedNewBlock.container.y = localPos.y + this.clickOffset[1];
        }
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
  
      if (!this.sidebar){
        maxWidth = util.updateMax(maxWidth, this.inputs.update());
        maxWidth = util.updateMax(maxWidth, this.outputs.update());
      }

      maxWidth = util.updateMax(maxWidth, this.blockBody.container.getBounds().width);
      
      this.blockBody.update(maxWidth + 2 * constants.TEXT_PADDING_LR);
    
      // after the maxWidth has been computed, position everything
      
      // position inputs
      this.inputs.container.x = (maxWidth - this.inputs.container.getBounds().width) / 2;
      this.inputs.container.y = constants.TEXT_PADDING_UD + constants.IO_PADDING_UD + constants.IO_TEXT_PADDING_UD * 5;
  
      // position outputs
      this.outputs.container.x = (maxWidth - this.outputs.container.getBounds().width) / 2;
      this.outputs.container.y = -constants.TEXT_PADDING_UD - constants.IO_PADDING_UD - constants.IO_TEXT_PADDING_UD * 3;
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
        100, 100, 100, 100);
        
      // create text
      if (props.isSidebar) {
        this.blockType = new Text(props.type, "40px Inter", props.color);
      } else {
        this.completeText();
        
      }
  
      this.container.addChild(this.blockType); 

      if (props.isSidebar) {
        this.shape.graphics.beginFill("#fff");
        this.shape.graphics.beginStroke("#fff");
      }

      this.shape.graphics.append(this.rect);
    }
  
    completeText = () => {
      console.log("COMPLETE TEXT:" + this.props.name + this.props.color)
      this.container.removeChild(this.blockType)
      this.blockName = new Text(this.props.name, "40px Inter", this.props.color);
      this.blockType = new Text(this.props.type + " :: ", "40px Inter", "#000");
      this.blockName.x = this.blockType.getBounds().width;
      this.container.addChild(this.blockType);
      this.container.addChild(this.blockName);
      this.rect.radiusTL = 100;
      this.rect.radiusTR = 100;
      this.rect.radiusBR = 100;
      this.rect.radiusBL = 100;
    }
  
    update = (width: number) => {
  
      let bounds: Rectangle = this.container.getBounds();

      let totalTextWidth: number = this.blockType.getBounds().width;
      if (this.blockName) {
        // if this is a block in the sidebar we won't have a block name
        totalTextWidth += this.blockName.getBounds().width;
      }

      this.rect.w = width;
      if (this.isSidebar) {
        this.rect.h = bounds.height;
      }else{
        this.rect.h = bounds.height + constants.TEXT_PADDING_UD * 2;
        this.rect.y = -constants.TEXT_PADDING_UD - 3;
      }

      this.blockType.x = (width - 2 * constants.TEXT_PADDING_LR - totalTextWidth) / 2;

      if (this.blockName){
        this.blockName.x = this.blockType.getBounds().width + (width - 2 * constants.TEXT_PADDING_LR - totalTextWidth) / 2;
      }
    }
  }