import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as constants from "./constants";
import {Block} from './block';

export class Sidebar {
    container: Container;
    newConv: Block;
    newAttentive: Block;
    shape: Shape;
    rect: Graphics.RoundRect;
    newConvXOffset: number = 20 + constants.TEXT_PADDING_LR;
    newConvYOffset: number = 100;
    newAttentiveXOffset: number = 20 + constants.TEXT_PADDING_LR;
    newAttentiveYOffset: number = 180;
  
    // have buttons be Blocks so that pressmove can work. on mousbuttondown add a new block to the sidebar.
    // blocks should have a sidebar state where they look different.
    constructor(stage: Stage) {
      this.container = new Container();
      this.container.x = 100;
      this.container.y = 100;
  
      this.shape = new Shape();
      this.shape.graphics.beginStroke("#ddd");
      this.shape.graphics.setStrokeStyle(5);
      this.shape.graphics.beginFill("#efefef");
      this.rect = new Graphics.RoundRect(0, 0, 220, 600, 50, 50, 50, 50);
      this.shape.graphics.append(this.rect);
      this.container.addChild(this.shape);
      stage.addChild(this.container);
  
      this.newConv = new Block(stage, this.container.x + this.newConvXOffset, this.container.y + this.newConvYOffset, 
          "#5B60E0", ["weights", "input", ], ['output'], "conv_1", "convolution", this);
      this.newAttentive = new Block(stage, this.container.x + this.newAttentiveXOffset, this.container.y + this.newAttentiveYOffset, 
          "#F97979", ["canvas", "intensity", "gx", "gy", "stride", "variance"], ['output2'], "write", "attentive_write", this);
    }
  }
  