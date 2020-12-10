import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {EditorElement} from './editorElement';
import { Output } from "./outputs";
import { Block } from "./block";

import * as util from './util'

export class Input extends EditorElement{
    container: Container;
    inputName: Text;
    shape: Shape; 
    rect: Graphics.RoundRect;

    fill: Graphics.Fill;

    connection: Output;
    focusedOutput: Output;

    background: String;
    textColor: String;

    params: any; 
  
    constructor(props) {
      super(props);
      this.container = new Container();
  
      this.inputName = new Text(props.name, "20px Inter", "#000");
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#000");
      this.shape.graphics.setStrokeStyle(2);

      this.container.addChild(this.shape);
      this.container.addChild(this.inputName);
  
      this.rect = new Graphics.RoundRect(-constants.IO_TEXT_PADDING_LR, -constants.IO_TEXT_PADDING_UD - 1,
                                         0, 0, 
                                         50, 50, 50, 50)
      this.shape.graphics.append(this.rect);

      this.params = props.params;
  
      if (this.props.availableParams) {
        this.fill = new Graphics.Fill("#222");
        this.inputName.color = "#eee";
      }else{
        this.fill = new Graphics.Fill("#fff");
      }

      this.shape.graphics.append(this.fill);
      
      this.update();
  
      this.container.addEventListener("mouseover", (event) => {
        if (!this.connection) {
          if (this.props.availableParams) {
            this.fill.style = "#555"
          }else{
            this.fill.style = '#ccc';
          }
        }else{
          this.inputName.color = "#fff";
        }
      });
  
      this.container.addEventListener("mouseout", (event) => {
        this.inputName.color = "#000";
        if (!this.connection) {
          
          if (this.props.availableParams) {
            this.fill.style = "#222";
            this.inputName.color = "#eee";
          }else{
            this.fill.style= "#fff";
          }
        }else{
          this.inputName.color = "#fff";

        }
        
      })

      this.container.addEventListener("mousedown", (event) => {
        this.focusedOutput = this.connection;
        if (this.params) {
          util.updateParameterEditor(this);
        }
      })

      this.container.addEventListener("pressmove", (event) => {
        if (this.connection) {
          console.log("REMOVING");
          this.remove();
        }

        if (this.focusedOutput) {
          this.focusedOutput.updateConnection(event.stageX, event.stageY);
        }
      })

      this.container.addEventListener("pressup", (event) => {
        if (this.focusedOutput) {
          this.focusedOutput.handleUp(event);
        }
      })
  
    }

    remove() {
      // remove connection Output-side
      this.connection.props.remove(this);
      this.connection = null;

      //remove connection Input-side
      this.props.remove(this);

    }
  
    update() {
      let bounds: Rectangle = this.inputName.getBounds();
  
      this.rect.w = bounds.width + 2 * constants.IO_TEXT_PADDING_LR;
      this.rect.h = bounds.height + 2 * constants.IO_TEXT_PADDING_UD;
  
      this.container.setBounds(0, 0, this.rect.w, this.rect.h);
    }
  }
  
export class Inputs extends EditorElement{
    container: Container;
    inputs: Input[] = [];

    children: [Output, Input][];    // these are the outputs that connect to a certain input. in other words, 
                                    // this Input depends on the following outputs
                                    // each input should only have one corresponding output
  
    constructor(props) {
      super(props);
      this.container = new Container();
      this.children = [];
  
      props.inputs.forEach((i: any, idx: number) => {
        let inp: Input = new Input({name: i.name, params: props.params, availableParams: i.params, blocks: props.blocks, remove: this.removeChild, block: props.block});
        this.inputs.push(inp);
        this.container.addChild(inp.container);
      });
  
      this.update();
    }
  //burver.
    addChild(pair: [Output, Input]): boolean {
      if (pair[1].connection != null) {
        // there is already an output connected to this input, so we'll disconnect the original output 
        return false;
      }

      this.children.push(pair);
      pair[1].connection = pair[0];

      return true
    }

    removeChild = (single?: Input) => {
      for (let i = 0; i < this.children.length; i++) {
        if (this.children[i][1] == single) {
          this.children.splice(i, 1);
          break;
        }
      }
    }

    updateConnections() {
      let seen: Set<Block> = new Set();
      this.children.forEach((pair: [Output, Input], idx: number) => {
        if (!seen.has(pair[0].props.block)){
          pair[0].props.block.outputs.updateConnections();
          seen.add(pair[0].props.block);
        }
      });
    }

    update(): number {
      let accWidth = 0;
      this.inputs.forEach((inp: Input, idx: number) => {
        inp.update();
        inp.container.x = accWidth;
        accWidth += inp.container.getBounds().width  + constants.IO_PADDING_LR;
      });
  
      return accWidth;
    }
  }