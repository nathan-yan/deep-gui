import { Stage, Point, Rectangle, Ticker, Graphics, Container, Shape, Text } from "@createjs/easeljs";

import * as util from "./util";

const TEXT_PADDING_LR = 60;
const TEXT_PADDING_UD = 33;
const IO_PADDING_LR = 10;
const IO_PADDING_UD = 20;
const IO_TEXT_PADDING_LR = 15;
const IO_TEXT_PADDING_UD = 10;

// store all non-moving elements
var staticObjects = new Array();

class Element {
  props: any;

  constructor(props) {
    this.props = props;
  }
}

class BlockBody extends Element{
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
    this.rect = new Graphics.RoundRect(-TEXT_PADDING_LR, -10,
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
    this.rect.y = -TEXT_PADDING_UD - 3;
  }

  update = (width: number) => {
    console.log(width);

    let bounds: Rectangle = this.container.getBounds();

    this.rect.w = width;
    this.rect.h = bounds.height + TEXT_PADDING_UD * 2;

    let totalTextWidth: number = this.blockType.getBounds().width + this.blockName.getBounds().width;
    this.blockType.x = (width - 2 * TEXT_PADDING_LR - totalTextWidth) / 2;
    this.blockName.x = this.blockType.getBounds().width + (width - 2 * TEXT_PADDING_LR - totalTextWidth) / 2;
    
  }
}

class Input extends Element{
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

    this.rect = new Graphics.RoundRect(-IO_TEXT_PADDING_LR, -IO_TEXT_PADDING_UD - 1,
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

    this.rect.w = bounds.width + 2 * IO_TEXT_PADDING_LR;
    this.rect.h = bounds.height + 2 * IO_TEXT_PADDING_UD;

    this.container.setBounds(0, 0, this.rect.w, this.rect.h);
  }
}

class Inputs extends Element{
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
      accWidth += inp.container.getBounds().width  + IO_PADDING_LR;
    });

    return accWidth;
  }
}

class Output extends Element{
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

    this.rect = new Graphics.RoundRect(-IO_TEXT_PADDING_LR, -IO_TEXT_PADDING_UD - 1,
                                       0, 0, 
                                       50, 50, 50, 50);
         
    this.arrow = new Graphics.BezierCurveTo(0, 0, 0, 0, 0, 0)
    this.p1 = new Graphics.Circle(0, 0, 5)

    this.shape.graphics.append(this.rect);

    this.move = new Graphics.MoveTo(this.rect.w /2, -IO_TEXT_PADDING_UD - 1);
    this.shape.graphics.beginFill("#faf0");
    this.shape.graphics.append(this.move);
    this.shape.graphics.append(this.arrow);
    this.shape.graphics.append(this.p1);
    //this.shape.graphics.append(this.p2);


    this.container.addEventListener("pressmove", (event) => {
      this.move.x = -IO_TEXT_PADDING_LR + this.rect.w / 2;

      this.shape.graphics.endFill()

      let pt: Point = this.container.globalToLocal(event.stageX, event.stageY);
      pt.y += 5;
      pt.x += 5;
      this.arrow.x = pt.x;
      this.arrow.y = pt.y;
      this.arrow.cp1x = -IO_TEXT_PADDING_LR + this.rect.w / 2;
      this.arrow.cp1y = (pt.y + (-IO_TEXT_PADDING_UD - 1)) / 2;


      this.arrow.cp2x = pt.x;
      this.arrow.cp2y = (pt.y + (-IO_TEXT_PADDING_UD - 1)) / 2;

      
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

    this.rect.w = bounds.width + 2 * IO_TEXT_PADDING_LR;
    this.rect.h = bounds.height + 2 * IO_TEXT_PADDING_UD;

    this.container.setBounds(0, 0, this.rect.w, this.rect.h);
  }
}

class Outputs extends Element{
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
      accWidth += inp.container.getBounds().width  + IO_PADDING_LR;
    });

    return accWidth - IO_PADDING_LR;
  }
}

class Block {
  container: Container; 
  blockBody: BlockBody;
  inputs: Inputs;
  outputs: Outputs;
  sidebar: Sidebar;  // the sidebar, if any, that this block belongs to. display the compact sidebar version of this block

  clickOffset: [number, number];
  //outputs: Output;

  constructor(stage: Stage, x: number, y: number, color: String, inputs: String[], outputs: String[], name: String, type: String, sidebar: Sidebar) {
    this.sidebar = sidebar;

    this.container = new Container();
    this.container.x = x;
    this.container.y = y;
    
    this.blockBody = new BlockBody({
        color: color,
        type: type,
        name: name,
        isSidebar: sidebar != null
     });
    this.inputs = new Inputs({
      inputs: inputs
    });
    this.outputs = new Outputs({
      inputs: outputs,
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
          this.sidebar.newConv = new Block(stage,
               this.sidebar.container.x + this.sidebar.newConvXOffset, this.sidebar.container.y + this.sidebar.newConvYOffset,
              "#5B60E0", ["weights", "input", ], ['output'], "conv_1", "convolution", this.sidebar);
        }
        else if (this == this.sidebar.newAttentive) {
          this.sidebar.newAttentive = new Block(stage,
              this.sidebar.container.x + this.sidebar.newAttentiveXOffset, this.sidebar.container.y + this.sidebar.newAttentiveYOffset,
              "#F97979", ["canvas", "intensity", "gx", "gy", "stride", "variance"], ['output2'], "write", "attentive_write", this.sidebar);
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
      this.container.y = localPos.y + this.clickOffset[1];;
    })

    this.update();
  }

  checkConnection(event: MouseEvent, output: Output) {
    console.log("i am checking");

  }

  update = () => {
    // find the bounds of BlockBody
    let maxWidth: number = 0;

    if (!this.blockBody.isSidebar) {
      maxWidth = util.updateMax(maxWidth, this.inputs.update());
      maxWidth = util.updateMax(maxWidth, this.outputs.update());
      maxWidth = util.updateMax(maxWidth, this.blockBody.container.getBounds().width);
      this.blockBody.update(maxWidth + 2 * TEXT_PADDING_LR);
    }
    // after the maxWidth has been computed, position everything
    
    // position inputs
    this.inputs.container.x = (maxWidth - this.inputs.container.getBounds().width) / 2;
    this.inputs.container.y = TEXT_PADDING_UD + IO_PADDING_UD + IO_TEXT_PADDING_UD * 5;

    // position outputs
    this.outputs.container.x = (maxWidth - this.outputs.container.getBounds().width) / 2;
    this.outputs.container.y = -TEXT_PADDING_UD - IO_PADDING_UD - IO_TEXT_PADDING_UD * 3;
  }
}


function tickGenerator(stage: Stage) {
  return function tick(event: Event) {
    stage.update();
    
  }
}

// scale function
function scale(stage, zoom, zoomPt, staticObjects) {
  // set zoom bounds
  if (zoom > 1 && stage.scale < 5 || zoom < 1 && stage.scale > 0.2) {
    let localPos = stage.globalToLocal(zoomPt[0], zoomPt[1]);
    stage.regX = localPos.x;
    stage.regY = localPos.y;
    stage.x = zoomPt[0];
    stage.y = zoomPt[1];

    stage.scale *= zoom;

    staticObjects.forEach(object => {
      object[0].graphics.command.w /= zoom;
      object[0].graphics.command.h /= zoom;

      let objPos = stage.globalToLocal(object[1], object[2]);
      object[0].graphics.command.x = objPos.x;
      object[0].graphics.command.y = objPos.y;
      console.log(stage.scale);
    })

    stage.update();
  }
}

// pan function
function pan(stage, screen, staticObjects) {
  screen.addEventListener("mousedown", (event1) => {
    let initPos = [stage.x, stage.y];

    screen.addEventListener('pressmove', (event2) => {
      stage.x = initPos[0] + event2.stageX - event1.stageX;
      stage.y = initPos[1] + event2.stageY - event1.stageY;
      
      staticObjects.forEach(object => {
        let pos = stage.globalToLocal(object[1], object[2]);
        object[0].graphics.command.x = pos.x;
        object[0].graphics.command.y = pos.y;
      })      
    })
  })
}

function zoomButtons(stage, canvas, zoomIntensity) {
  // zoom buttons (might be better to replace with html buttons)
  let zoomIn = new Shape();
  zoomIn.graphics.beginFill("white").drawRect(25, 25, 50, 50);
  staticObjects.push([zoomIn, 25, 25]);
  stage.addChild(zoomIn);

  zoomIn.addEventListener("click", (event) => {
    scale(stage, zoomIntensity, [canvas.width/2, canvas.height/2], staticObjects);
  })

  let zoomOut = new Shape();
  zoomOut.graphics.beginFill("white").drawRect(25, 100, 50, 50);
  staticObjects.push([zoomOut, 25, 100]);
  stage.addChild(zoomOut);

  zoomOut.addEventListener("click", (event) => {
    scale(stage, 1/zoomIntensity, [canvas.width/2, canvas.height/2], staticObjects);
  })
}

class Sidebar {
  container: Container;
  newConv: Block;
  newAttentive: Block;
  shape: Shape;
  rect: Graphics.RoundRect;
  newConvXOffset: number = 20 + TEXT_PADDING_LR;
  newConvYOffset: number = 100;
  newAttentiveXOffset: number = 20 + TEXT_PADDING_LR;
  newAttentiveYOffset: number = 180;

  // have buttons be Blocks so that pressmove can work. on mousbuttondown add a new block to the sidebar.
  // blocks should have a sidebar state where they look different.
  constructor(stage: Stage) {
    this.container = new Container();
    this.container.x = 40;
    this.container.y = 40;

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


window.addEventListener("load", () => {
  //get the canvas, canvas context, and dpi
  
  let canvas = <HTMLCanvasElement> document.getElementById('myCanvas'),
  ctx = canvas.getContext('2d'),
  dpi = window.devicePixelRatio * 2;

  canvas.width = 2000;
  canvas.height = 1600;
  canvas.style.width = "1000px";
  canvas.style.height = "800px";
  ctx.scale(2, 2);

  //Create a stage by getting a reference to the canvas
  let stage = new Stage("myCanvas");
  stage.enableMouseOver(10);
  
  let sidebar = new Sidebar(stage)

  // set up a customizable background screen
  let screen = new Shape();
  screen.graphics.beginLinearGradientFill(["#CC91FF" ,"#91A9FF"], [0, 1], -2*canvas.width, -2*canvas.height, 2*canvas.width, 2*canvas.height).drawRect(0, 0, canvas.width, canvas.height);
  staticObjects.push([screen, 0, 0]);
  stage.addChild(screen);
  
  // change how much stage zooms each step
  let zoomIntensity = 1.2;

  // mouse wheel zoom
  canvas.addEventListener("wheel", (event) => {
    let zoom = event.deltaY < 0 ? 1/zoomIntensity : zoomIntensity;
    scale(stage, zoom, [stage.mouseX, stage.mouseY], staticObjects);
  });

  // click and drag pan
  pan(stage, screen, staticObjects);

  let block: Block = new Block(stage, 800, 400, "#5B60E0", ["weights", "input", ], ['output'], "conv_1", "convolution", null);
  let block2: Block = new Block(stage, 800, 100, "#F97979", ["canvas", "intensity", "gx", "gy", "stride", "variance"], ['output2'], "write", "attentive_write", null);
  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]); 

  // zoom buttons
  zoomButtons(stage, canvas, zoomIntensity);
  
  //let block2: Block = new Block(stage, 100, 100, "#5B60E0", ["input1", "test", ]);

  Ticker.framerate = 60;
  Ticker.addEventListener('tick', tickGenerator(stage));  
})
