import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {EditorElement} from './editorElement';

export class Input extends EditorElement{
    container: Container;
    inputName: Text;
    shape: Shape; 
    rect: Graphics.RoundRect;
  
    constructor(props) {
      super(props);
      this.container = new Container();
  
      this.inputName = new Text(props.name, "20px Inter", "#000");
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#000");
      this.shape.graphics.setStrokeStyle(2);
      this.shape.graphics.beginFill("#fff");
  
      this.container.addChild(this.shape);
      this.container.addChild(this.inputName);
  
      this.rect = new Graphics.RoundRect(-constants.IO_TEXT_PADDING_LR, -constants.IO_TEXT_PADDING_UD - 1,
                                         0, 0, 
                                         50, 50, 50, 50)
      this.shape.graphics.append(this.rect);
  
      
      this.update();
  
      this.container.addEventListener("mouseover", (event) => {
        this.inputName.color = "#faf";
      });
  
      this.container.addEventListener("mouseout", (event) => {
        this.inputName.color = "#000";
      })
  
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
  
    constructor(props) {
      super(props);
      this.container = new Container();
  
      props.inputs.forEach((value: String, idx: number) => {
        let inp: Input = new Input({name: value});
        this.inputs.push(inp);
        this.container.addChild(inp.container);
      });
  
      this.update();
    }
  //burver.
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