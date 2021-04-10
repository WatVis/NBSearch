// mouse over&out handler, used in alignment.js for handle different events
function handleMOver_square(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolSquare).size(180));
}

function handleMOver_circle(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolCircle).size(150));
}

function handleMout_square(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolSquare).size(60));
}

function handleMout_circle(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolCircle).size(50));
}



// Mouse_out effects
function handleMout_square_result(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolSquare).size(120));
}

function handleMout_circle_result(d, i) {
    d3.select(this)
        .attr("d", d3.symbol().type(d3.symbolCircle).size(100));
}
