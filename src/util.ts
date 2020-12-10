import {Input} from './inputs';

export function updateMax(v: number, n: number): number {
    if (n > v) {
        return n;
    }

    return v;
}

export function draw(elements: any[], container: any) {
    
}

export function updateParameterEditor(inp: Input) {
    console.log("event!");
    let type: HTMLSpanElement = document.getElementById("block-type");
    let name: HTMLSpanElement = document.getElementById("block-name");
    let inputName: HTMLSpanElement = document.getElementById("input-name");
    type.innerText = inp.props.block.blockBody.blockType.text;
    name.innerText = inp.props.block.blockBody.name;
    inputName.innerText = inp.props.name;

    let editor: HTMLDivElement = <HTMLDivElement> document.getElementById("parameters");
    editor.innerHTML = '';

    inp.props.availableParams.forEach((param: any, idx: number) => {
        let p = inp.props.params[param];

        let element: HTMLDivElement = <HTMLDivElement> document.createElement('div');
        element.className = 'input-parameter'
        element.innerHTML = `
            <span style = "">${param}</span>: <input id = input-${idx} class = 'inp' style = "color: ${inp.props.block.color}" value = ${p.value}></input> 
            <br/>
            <span style = 'color: #aaa'>${p.type}</span>
        `

        console.log(element);

        editor.appendChild(element);

        document.getElementById("input-" + idx).onchange = (e: any) => {
            inp.props.params[param].value = e.target.value;
        } 
    })

    let text: HTMLDivElement = <HTMLDivElement> document.createElement("div");
    text.style.marginTop = "50px";
    text.innerText = "Press ENTER to save";

    editor.appendChild(text);
    
}