import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import * as util from './util';

import {EditorElement} from './editorElement';
import {Inputs, Input} from './inputs';
import {Output, Outputs} from './outputs';
import {Sidebar} from './sidebar'

export class Block {
    stage: Container;

    container: Container; 
    blockBody: BlockBody;
    inputs: Inputs;
    outputs: Outputs;
    sidebar: boolean;  // the sidebar, if any, that this block belongs to. display the compact sidebar version of this block
  
    blocks: Block[] = [];   // the list of all other blocks, since this is pass by reference we chillin on the memory usage B)

    clickOffset: [number, number];
    //outputs: Output;

    children: [Input, Output][];
    color: String

    focusedNewBlock: Block;
    parameters: any; 
  
    constructor(stage: Container, parameters: any, blocks: Block[], x: number, y: number, color: String, inputs: any[], outputs: String[], name: String, type: String, sidebar: boolean, displayStage?: Container) {
      console.log(blocks);
      this.sidebar = sidebar;
      this.stage = stage;
  
      this.container = new Container();
      this.container.x = x;
      this.container.y = y;
      
      if (sidebar) {
        this.container.initialY;
      }
     
      this.parameters = parameters;

      this.blocks = blocks;
      this.color = color;
      
      this.children = [];
      
      this.blockBody = new BlockBody({
          color: color,
          type: type,
          name: name,
          isSidebar: sidebar == true,
          block: this
       });
      
      this.inputs = new Inputs({
        inputs: inputs,
        block: this,
        params: this.parameters
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
        let localPos = this.stage.globalToLocal(event.stageX, event.stageY);
        this.clickOffset = [this.container.x - localPos.x,
                                  this.container.y - localPos.y];

        if (this.sidebar) {
          let paramsClone;
          if (this.parameters) {
            paramsClone = JSON.parse(JSON.stringify(this.parameters))
          }else {
            paramsClone = {}
          }

            let newBlock: Block = new Block(this.stage, paramsClone, this.blocks,
                localPos.x + this.clickOffset[0], localPos.y + this.clickOffset[1],
                color,  inputs, outputs, name, type, false);
            
            this.blocks.push(newBlock);
            this.focusedNewBlock = newBlock;
          } 
        
      })

      this.blockBody.container.addEventListener("pressup", (event) => {
        if (this.focusedNewBlock && this.sidebar) {
          this.stage.removeChild(this.focusedNewBlock.container);
          
          let pt: Point = displayStage.globalToLocal(this.focusedNewBlock.container.x, this.focusedNewBlock.container.y);

          this.focusedNewBlock.container.x = pt.x;
          this.focusedNewBlock.container.y = pt.y;

          displayStage.addChild(this.focusedNewBlock.container);
          this.focusedNewBlock.stage = displayStage;
        }
      })
  
      this.blockBody.container.addEventListener('pressmove', (event) => {
        let localPos: Point = this.stage.globalToLocal(event.stageX, event.stageY);

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
              if(!block.inputs.addChild([output, inp])) {
                inp.remove();
                block.inputs.addChild([output, inp])
              }

              this.outputs.addParent([output, inp]);
              inp.fill.style = this.color;
              inp.inputName.color = "#fff";
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
    name: string;

    otherWidths: number; // the width of all the other elements
                         // if block name + block type is less than otherWidths, the rounded rectangle
                         // is as wide as otherWidths. if block name + block type is longer,
                         // the rounded rectangle will enlarge to fit the name and type

    selectionRect: Graphics.Rect;

    constructor(props) {
      super(props);
  
      this.isSidebar = props.isSidebar;
  
      this.container = new Container();
      this.shape = new Shape();

      this.shape.graphics.beginStroke(props.color);

      this.shape.graphics.setStrokeStyle(5);
      this.shape.graphics.beginFill("#fff");
  
      this.container.addChild(this.shape);

      this.name = props.name;
  
      if (props.isSidebar) {
        this.shape.graphics.beginFill("#fff");
        this.shape.graphics.beginStroke("#fff");
      }
      // x, y, w, h, corners x 4
      // default values here are for sidebar display
      this.rect = new Graphics.RoundRect(-constants.TEXT_PADDING_LR, -10,
        180, 60,
        100, 100, 100, 100);
      
      this.shape.graphics.append(this.rect);

      this.shape.graphics.setStrokeStyle(1).beginStroke("#222");
      this.shape.graphics.beginFill("#eee");
      this.selectionRect = new Graphics.Rect(0, 0, 0, 0);

      this.shape.graphics.append(this.selectionRect);

      
      // create text
      if (props.isSidebar) {
        this.blockType = new Text(props.type, "40px Inter", props.color);
      } else {
        this.completeText();
        
      }
  
      this.container.addChild(this.blockType); 

      


      if (!props.isSidebar) {
        this.blockName.addEventListener("mousedown", (event) => {
          let inputElement: HTMLInputElement = <HTMLInputElement> document.getElementById("name-editor");
          inputElement.value = this.name;
          inputElement.oninput = (event: any) => {
            console.log(event.target.selectionStart);
            this.updateName(event.target.value);
          }
          inputElement.onkeydown = (event: any) => {
            setTimeout(() => {
              console.log(event.target.selectionStart + " " + event.target.selectionEnd)
              this.updateSelection(event.target.selectionStart, event.target.selectionEnd);
            }, 10);
          }
          inputElement.focus();
          
        })
      }
    }

    updateName = (name: string) => {
      this.name = name;
      this.blockName.text = name;
      this.props.block.update();
    }

    updateSelection = (start: number, end: number) => {
      
      let temp: Text;
      let offsetX: number;

      if (start == 0) {
        temp = new Text("a", "40px Inter");
        offsetX = 0;
      }else {
        temp = new Text(this.name.slice(0, start), "40px Inter");

        offsetX = temp.getBounds().width
      }

      let width: number;
      let height: number;
      
      if (start == end) {
        width = 1;
        height = temp.getBounds().height - 5;  
      } else {
        temp.text = this.name.slice(start, end);
        width = temp.getBounds().width;
        height = temp.getBounds().height - 5;
      }

      this.selectionRect.x = offsetX + this.blockName.x;
      this.selectionRect.y = 0;
      this.selectionRect.w = width;
      this.selectionRect.h = height;
    }
  
    completeText = () => {
      console.log("COMPLETE TEXT:" + this.props.name + this.props.color)
      this.container.removeChild(this.blockType)
      this.blockName = new Text(this.name, "40px Inter", this.props.color);
      this.blockType = new Text(this.props.type + " :: ", "40px Inter", "#000");
      this.blockName.x = this.blockType.getBounds().width;
      this.container.addChild(this.blockType);
      this.container.addChild(this.blockName);
      this.rect.radiusTL = 100;
      this.rect.radiusTR = 100;
      this.rect.radiusBR = 100;
      this.rect.radiusBL = 100;
    }
  
    update = (width?: number) => {
      if (width) {
        this.otherWidths = width;
      }else {
        width = this.otherWidths;
        width = util.updateMax(width, this.container.getBounds().width + 2 * constants.TEXT_PADDING_LR)
      }

      let bounds: Rectangle = this.container.getBounds();

      let totalTextWidth: number = this.blockType.getBounds().width;
      if (this.blockName) {
        // if this is a block in the sidebar we won't have a block name
        totalTextWidth += this.blockName.getBounds().width;
      }

      this.rect.w = width;
      if (this.isSidebar) {
        this.rect.w -= constants.TEXT_PADDING_LR * 2;
        this.rect.h = bounds.height;
        this.rect.x = 0;
        this.rect.y = 2;
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