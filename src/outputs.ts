
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
  
    connections;
  
    constructor(props) {
      super(props);
      this.container = new Container();
      this.connections = {};
  
      this.inputName = new Text(props.name, "20px Inter", "#000");
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#000");
      this.shape.graphics.setStrokeStyle(2);
  
      this.container.addChild(this.shape);
      this.container.addChild(this.inputName);
  
      this.rect = new Graphics.RoundRect(-constants.IO_TEXT_PADDING_LR, -constants.IO_TEXT_PADDING_UD - 1,
                                         0, 0, 
                                         50, 50, 50, 50);
           
      this.arrow = new Graphics.BezierCurveTo(0, 0, 0, 0, 0, 0)
      //this.p1 = new Graphics.Circle(0, 0, 5)
  
  
      this.move = new Graphics.MoveTo(this.rect.w /2, -constants.IO_TEXT_PADDING_UD - 1);
      this.shape.graphics.beginFill("#0000");

      this.shape.graphics.append(this.move);
      this.shape.graphics.append(this.arrow);
      //this.shape.graphics.append(this.p1);
      //this.shape.graphics.append(this.p2);
      this.shape.graphics.beginFill("#fff");

      this.shape.graphics.append(this.rect);

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
  
        
        //this.p1.x = pt.x + 2.5;
        //this.p1.y = pt.y - 2.5;
  
        /*
        this.p2.x = this.arrow.cp2x;
        this.p2.y = this.arrow.cp2y;
        */
      })
  
      this.container.addEventListener("pressup", (event) => {
        console.log("clicked up! ")
        if(!this.props.checkConnection(event, this)) {
          this.arrow.x = 0;
          this.arrow.y = 0;
          this.arrow.cp1x = 0;
          this.arrow.cp2x = 0;
          this.arrow.cp1y = 0;
          this.arrow.cp2y = 0;
        } 
        
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

    parents: [Output, Input][];
    arrows: [Graphics.MoveTo, Graphics.BezierCurveTo][];

    arrowShapes: Shape;  

    constructor(props) {
      super(props);
      this.container = new Container();

      this.arrows = [];
      this.arrowShapes = new Shape();
      this.arrowShapes.graphics.beginStroke("#000");
      this.arrowShapes.graphics.setStrokeStyle(3);

      this.parents = [];
  
      props.inputs.forEach((value: String, idx: number) => {
        let inp: Output = new Output({name: value, checkConnection: this.props.checkConnection, block: this.props.block});
        this.inputs.push(inp);
        this.container.addChild(inp.container);
      });

      this.container.addChild(this.arrowShapes);
  
      this.update();
    }

    addParent = (pair: [Output, Input]) => {
      // create a new moveTo and a new bezier curve
      let out: Output = pair[0];
      let inp: Input = pair[1];

      this.parents.push(pair);
      
      let move: Graphics.MoveTo = new Graphics.MoveTo(inp.container.x + inp.rect.w /2, -constants.IO_TEXT_PADDING_UD - 1);
      let arrow: Graphics.BezierCurveTo = new Graphics.BezierCurveTo(0, 0, 0, 0, 0, 0);

      this.arrows.push([move, arrow]);
      this.arrowShapes.graphics.append(move);
      this.arrowShapes.graphics.append(arrow);

      this.updateConnections();
    }

    update(): number {
      let accWidth = 0;
      this.inputs.forEach((inp: Output, idx: number) => {
        inp.update();
        inp.container.x = accWidth;
        accWidth += inp.container.getBounds().width  + constants.IO_PADDING_LR;
      });

      return accWidth - constants.IO_PADDING_LR;
    }

    updateConnections() {
      this.arrows.forEach((graphic: [Graphics.MoveTo, Graphics.BezierCurveTo], idx: number) => {
        graphic[0].x = this.parents[idx][0].container.x + this.parents[idx][0].rect.w / 2 - constants.IO_TEXT_PADDING_LR;

        let inp: Input = this.parents[idx][1];
        let out: Output = this.parents[idx][0];

        let global: Point = inp.container.localToGlobal(inp.rect.w / 2 - constants.IO_TEXT_PADDING_LR, inp.rect.h / 2 + constants.IO_TEXT_PADDING_UD);
        let local: Point = this.container.globalToLocal(global.x, global.y);

        graphic[1].x = local.x;
        graphic[1].y = local.y;
        graphic[1].cp1x = -constants.IO_TEXT_PADDING_LR + out.rect.w / 2;
        graphic[1].cp1y = (local.y + (-constants.IO_TEXT_PADDING_UD - 1)) / 2;
  
        graphic[1].cp2x = local.x;
        graphic[1].cp2y = (local.y + (-constants.IO_TEXT_PADDING_UD - 1)) / 2;
      })
    }
  }
  