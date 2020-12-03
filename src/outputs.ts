
import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {EditorElement} from './editorElement';

import {Input} from './inputs';

export class Output extends EditorElement{
    container: Container;
    inputName: Text;
    shape: Shape; 
    rect: Graphics.RoundRect;
  
    arrow: Graphics.BezierCurveTo;
    move: Graphics.MoveTo;
  
    p1: Graphics.Circle;
    p2: Graphics.Circle;
  
    parents: Input[];
  
    constructor(props) {
      super(props);
      this.container = new Container();
  
      this.inputName = new Text(props.name, "20px Inter", "#000");
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#000");
      this.shape.graphics.setStrokeStyle(2);
      this.shape.graphics.beginFill("#aff");
  
      this.container.addChild(this.shape);
      this.container.addChild(this.inputName);
  
      this.rect = new Graphics.RoundRect(-constants.IO_TEXT_PADDING_LR, -constants.IO_TEXT_PADDING_UD - 1,
                                         0, 0, 
                                         50, 50, 50, 50);
           
      this.arrow = new Graphics.BezierCurveTo(0, 0, 0, 0, 0, 0)
      this.p1 = new Graphics.Circle(0, 0, 5)
  
      this.shape.graphics.append(this.rect);
  
      this.move = new Graphics.MoveTo(this.rect.w /2, -constants.IO_TEXT_PADDING_UD - 1);
      this.shape.graphics.beginFill("#faf0");
      this.shape.graphics.append(this.move);
      this.shape.graphics.append(this.arrow);
      this.shape.graphics.append(this.p1);
      //this.shape.graphics.append(this.p2);
  
  
      this.container.addEventListener("pressmove", (event) => {
        this.move.x = -constants.IO_TEXT_PADDING_LR + this.rect.w / 2;
  
        this.shape.graphics.endFill()
  
        let pt: Point = this.container.globalToLocal(event.stageX, event.stageY);
        pt.y += 5;
        pt.x += 5;
        this.arrow.x = pt.x;
        this.arrow.y = pt.y;
        this.arrow.cp1x = -constants.IO_TEXT_PADDING_LR + this.rect.w / 2;
        this.arrow.cp1y = (pt.y + (-constants.IO_TEXT_PADDING_UD - 1)) / 2;
  
  
        this.arrow.cp2x = pt.x;
        this.arrow.cp2y = (pt.y + (-constants.IO_TEXT_PADDING_UD - 1)) / 2;
  
        
        this.p1.x = pt.x + 2.5;
        this.p1.y = pt.y - 2.5;
  
        /*
        this.p2.x = this.arrow.cp2x;
        this.p2.y = this.arrow.cp2y;
        */
      })
  
      this.container.addEventListener("pressup", (event) => {
        console.log("clicked up! ")
        this.props.checkConnection(event, this);
      })
  
      this.update();
    }
  
    update() {
      let bounds: Rectangle = this.inputName.getBounds();
  
      this.rect.w = bounds.width + 2 * constants.IO_TEXT_PADDING_LR;
      this.rect.h = bounds.height + 2 * constants.IO_TEXT_PADDING_UD;
  
      this.container.setBounds(0, 0, this.rect.w, this.rect.h);
    }
  }
  
export class Outputs extends EditorElement{
    container: Container;
    inputs: Output[] = [];
  
    constructor(props) {
      super(props);
      this.container = new Container();
  
      props.inputs.forEach((value: String, idx: number) => {
        let inp: Output = new Output({name: value, checkConnection: this.props.checkConnection});
        this.inputs.push(inp);
        this.container.addChild(inp.container);
      });
  
      this.update();
    }
  //burver.
    update(): number {
      let accWidth = 0;
      this.inputs.forEach((inp: Output, idx: number) => {
        inp.update();
        inp.container.x = accWidth;
        accWidth += inp.container.getBounds().width  + constants.IO_PADDING_LR;
      });
  
      return accWidth - constants.IO_PADDING_LR;
    }
  }
  