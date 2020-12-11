import {Inputs, Input} from './inputs';

export class ParameterEditor{
    element: HTMLElement;

    ParameterEditor(id) {
        // attach onto the id
        this.element = document.getElementById(id);
    }

    update(inp: Input) {
        if (inp.params) {
            inp.params.forEach((param: any, idx: number) => {
                
            })
        }
    }
}
