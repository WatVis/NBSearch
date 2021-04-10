// READ LIST_VIEW.JS FIRST, INTERACTIONS THERE ARE EASIER TO UNDERSTAND

///////////////// This file contains rendering of alignment_view, and some necessary interactions are being set up here. //////////
// Interactions includes demonstrating detail_view, showing arrows et al.. ////////////////////////////////////////////////////////

/////// global variables //////////
var curr_arrow_nb = 0;
var curr_arrow_cell = 0;
var shifted = true;
var arrow_order = [];
///////////////////////////////////

// set up the tool for drawing paths in alignment view
var areaGenerator = d3.area();
areaGenerator.x0(function (d) {
    return d.left
}).x1(function (d) {
    return d.right
}).y(function (d) {
    return d.y
});
areaGenerator
    .defined(function (d) {
        return d !== null;
    });

// Actually generate the path, and then append it to .line_layer
for (let j = 0; j < align_path_data.length; j++) {
    // generate path
    var a_path = areaGenerator(align_path_data[j]);
    // append it to .line_layer
    d3.select('.line_layer')
        .append('path')
        .attr('d', a_path)
        .style('fill', function (d, i) {
            return color[j]; // wrap index around colour
        }).style('opacity', 0.4);
}

// set up tool for drawing rectangles
var squareGenerator = d3.symbol()
    .type(d3.symbolSquare)
    .size(60);
var squareData = squareGenerator();

// draw the rectangles, and set up attributes, as well as OnClick event handler.
d3.select('.cell_layer')
    .selectAll('squares')
    .data(align_start_pts)
    .enter()
    .append('path')
    .attr('class', 'start_cell')
    .attr('transform', function (d) {
        // d should be [x, y]
        return 'translate(' + d + ')';
    })
    .attr('d', function (d, i) {
        // return the tool we said before.
        // return another one if it's the special case.
        // This defines the shape & size of our drawing method
        if (emb_ratio[i][real_index[i][0]] === 1) {
            const SG = d3.symbol()
                .type(d3.symbolSquare)
                .size(130);
            return SG();
        }
        return squareData;
    })
    .style('opacity', function (d, i) {
        if (emb_ratio[i][0] === 1) {
            return 1;
        } else {
            return 0.65
        }
    })
    .style('fill', function (d, i) {
        return color[i];
    })
    .attr('id', function(d, i) {
        return 'cell' + i.toString() + '0';
    })
    .on('click', function (d, i) {
        // set the global variable (current selected cell)
        curr_selected_cell = d3.select('#cell' + i.toString() + '0');
        // first lets deal with alignment view
        d3.selectAll('.start_cell')
            .style('stroke', 'none');
        d3.selectAll('.middle_cell')
            .style('stroke', 'none');
        d3.select(this)
            .style('stroke', 'black')
            .style('stroke-width', '2px');


        // first remove innerHTML of detail_view
        document.getElementById('detail_view_body').innerText = "";
        document.getElementById('arrow').innerText = "";
        // remove other arrow_links
        d3.selectAll('.arrow_link').remove();


        //////////////////////// arrwos are appended in this section /////////////
        let curr_slider_value = document.getElementById('myRange').value;
        var arrowlineGenerator = d3.line()
            .curve(d3.curveNatural);

        var arrow_points = shift_arrow_path[i][0];

        if (shifted === false) {
            arrow_points = unshift_arrow_path[i][0];
        }

        arrow_order = [];

        for (var ct = 0; ct < arrow_points.length; ct++) {
            curr_arrow_nb = i;
            curr_arrow_cell = 0;
            let the_obj = arrow_points[ct];
            if (the_obj['count'] >= curr_slider_value) {
                // do it
                arrow_order.push(ct);
                d3.select('.arrow_layer')
                    .append('path')
                    .attr('class', 'arrow_link')
                    .attr('d', arrowlineGenerator(the_obj['position'][0]));
            }
        }

        //////////////////// end of adding arrows ////////////////////////////////

        // then append cells into detail_view
        var detail_div = d3.select('#detail_view_body');
        var number_of_cells = details_data[i].length;
        var detail_cell_width = 460;
        var detail_cell_height = 100;

        // Before appending the first cell, make tags ready.
        // TAG preparation ////////////////////////////////////////////////////////////////////
        // 1. found the variable name corpus
        const var_name_corpus = all_variable_names[i][0];
        // 2. found specific variables
        //    notice that, the above corpus contains the below variables
        //    Also, we need to record linked cells
        var linked_cells = [];
        var linked_cell_vars = [];
        const curr_cell_info = unshift_arrow_path[i][0];
        var used_name_arr = [];
        if (curr_cell_info !== undefined) {
            for (var count = 0; count < curr_cell_info.length; count++) {
                // if line was drawn, then I will consider its variables
                //    as 'used'
                if (curr_cell_info[count]['count'] >= curr_slider_value) {
                    used_name_arr = used_name_arr.concat(curr_cell_info[count]['var_names']);
                    linked_cells.push(curr_cell_info[count]['index']);
                    linked_cell_vars.push(curr_cell_info[count]['var_names']);
                }
            }
            const unique_set = new Set(used_name_arr);
            used_name_arr = [...unique_set];
        }
        // 3. current searched_cell variables:
        const curr_searched_cell_var = store_searched_cell_vars[i];
        // end of TAG preparation ///////////////////////////////////////////////////////////////


        // start appending the cell
        for (let j = 0; j < number_of_cells; j++) {
            // call 'detail_view.append(...)' in a for loop will keep adding child to detail_view
            var a_container = detail_div.append('div')
                .attr('class', 'the_containers')
                .attr('id', 'container' + j.toString())
                .style('top', (j * (5)).toString() + 'px');

            // change the style based on ratio. If ratio = 1, make it different than others
            if (emb_ratio[i][j] === 1) {
                a_container.style('border', '2px solid')
                    .style('border-color', color[i]);
            } else {
                a_container.style('border', '1px solid');
            }
            // append progress bar
            var a_bar = a_container.append('div')
                .attr('class', 'progress_bar')
                .attr('id', 'the_bar' + j.toString())
                .style('width', (detail_cell_width * emb_ratio[i][j]).toString() + 'px')
                .style('background-color', color[i]);


            // Add TAGGGGGGGGGGGGG
            if (j === real_index[i][0]) {
                // OK this is the corresponding cell user clicked
                // if nothing, then it's meaningless to do that.
                if (var_name_corpus !== undefined) {
                    if (var_name_corpus.length > 0) {

                        var the_tag_div = a_container.append('div')
                            .attr('class', 'tag_container')
                            .attr('id', 'the_tag_div' + j.toString());
                        // change the size of biggest container
                        //a_container.style('height', (store_detail_height[i][j] + 43).toString() + 'px');
                        for (var loop = 0; loop < var_name_corpus.length; loop++) {
                            const the_name = var_name_corpus[loop];
                            const if_includes = used_name_arr.includes(the_name);
                            const if_searched_cell_var = curr_searched_cell_var.includes(the_name);
                            the_tag_div.append('div')
                                .attr('class', 'word_tag')
                                .attr('id', 'tag' + loop.toString())
                                .style('left', ((loop + 1) * 6).toString() + 'px')
                                .text(the_name)
                                .style('background-color', function(d, k) {
                                    if (if_includes) {
                                        return '#c7c7c7';
                                    }
                                    return 'white';
                                })
                                .style('border', function(d, k) {
                                    if (if_searched_cell_var) {
                                        return '2px solid';
                                    }
                                });
                        }
                    }
                } else {
                    // do nothing
                }
            } else if (linked_cells.includes(j)) {
                // linked cells, these 2 cells have shared variables
                var a_tag_div = a_container.append('div')
                    .attr('class', 'tag_container')
                    .attr('id', 'the_tag_div' + j.toString());
                // change the size of biggest container
                const curr_var_name_corpus = all_variable_names[i][j];
                for (var looop = 0; looop < curr_var_name_corpus.length; looop++) {
                    const the_name = curr_var_name_corpus[looop];
                    // find correct position
                    const posi = linked_cells.findIndex((element) => element === j);
                    const if_inc = linked_cell_vars[posi].includes(the_name);
                    const if_search_cell_var = curr_searched_cell_var.includes(the_name);
                    // keep adding word_tag onto tag_container
                    a_tag_div.append('div')
                        .attr('class', 'word_tag')
                        .attr('id', 'tag' + looop.toString())
                        .style('left', ((looop + 1) * 6).toString() + 'px')
                        .text(the_name)
                        .style('background-color', function(d, k) {
                            if (if_inc) {
                                return '#c7c7c7';
                            }
                            return 'white';
                        })
                        .style('border', function(d, k) {
                            if (if_search_cell_var) {
                                return '2px solid';
                            }
                        });
                }
            }
            // end of adding TAGGGGGGGGGGG

            // Starting appending fake code cell, similar to what done in list_view.js
            var a_detail_text = a_container.append('div')
                .attr('class', 'fake_textarea detail_view_text')
                .attr('id', 'detail_view_text' + j.toString())
                .attr('disabled', 'true')
                .append('pre')
                .append('code') // check list_view.js for reasons added 'pre' & 'code'
                .attr('id', 'detail_fake_code' + j.toString())
                .attr('class', 'python hljs');
            // for some reason, the above 'python hljs' doesnt seem to work, and
            //    we have to call hljs.highlight(...) manually here
            document.getElementById('detail_fake_code' + j.toString()).innerHTML =
                hljs.highlight('python', details_data[i][j].conc_cell, 'true').value;

        }

        var true_index = real_index[i][0];
        console.log(true_index);

        // we scroll to that position, then make it different for a while
        var myElement = document.getElementById('container' + true_index.toString());
        document.getElementById('detail_view').scrollTop = myElement.offsetTop;

        // scroll to proper position of list_view
        var myElement2 = document.getElementById('list_cell' + i.toString());
        document.getElementById('list_view').scrollTop = myElement2.offsetTop;

        // Add an arrow to let user find the cell easily.
        d3.select('#arrow')
            .append('p')
            .attr('id', 'detail_left_arrow')
            .style('position', 'absolute')
            .style('top', myElement.offsetTop.toString() + 'px');
        document.getElementById('detail_left_arrow').innerText = '←';
    })
    .on('mouseover', handleMOver_square) // check alignment_handler.js
    .on('mouseout', function (d, i) {
        if (emb_ratio[i][0] === 1) {
            d3.select(this).attr("d", d3.symbol().type(d3.symbolSquare).size(130));
        } else {
            d3.select(this).attr("d", d3.symbol().type(d3.symbolSquare).size(60));
        }
    });
// The above code renders first cells of all notebooks as rectangles, and also set up
// some proper OnClick reactions.


// Here, we create circles for middle points, the reaction is very similar to the above,
// But since data is now 2D (previously, for example, all fisrt cell of 10 notebooks is just a list of 10)
//     indexes are a little bit hard to remember and imagine...

// set up tool for drawing circles
var circleGenerator = d3.symbol()
    .type(d3.symbolCircle)
    .size(50);
var circleData = circleGenerator();

// check line 47.
// similar here, each iteration we draw a group (column) of nodes
for (let k = 0; k < align_middle_pts.length; k++) {
    d3.select('.cell_layer')
        .selectAll('circles')
        .data(align_middle_pts[k]) // draw the middle points of the k-th notebooks
        .enter()
        .append('path')
        .attr('class', 'middle_cell')
        .attr('transform', function (d) {
            // d should be [x, y]
            return 'translate(' + d + ')';
        })
        .attr('d', function (d, i) {
            // return value is the drawing method, i.e. denifition of shape & size.
            if (emb_ratio[k][real_index[k][i + 1]] === 1) {
                const SG = d3.symbol()
                    .type(d3.symbolCircle)
                    .size(120);
                return SG();
            }
            return circleData;
        })
        .style('opacity', function (d, i) {
            if (emb_ratio[k][i + 1] === 1) {
                return 1;
            } else {
                return 0.65;
            }
        })
        .style('fill', function (d, i) {
            return color[k];
        })

        .attr('id', function(d, i) {
            return 'cell' + k.toString() + (i + 1).toString();
        })
        .on('click', function (d, i) {
            ///////////////////////////////////////////////////////////////////////////
            // WHEN ENTER THIS FUNCTION:
            // USER clicked k-th notebook, i-th MIDDLE node. k is defined at line 319
            ///////////////////////////////////////////////////////////////////////////
            curr_selected_cell = d3.select('#cell' + k.toString() + (i + 1).toString());
            // (i + 1) == j
            var true_index = real_index[k][i + 1];


            //////////////////// STARTING FROM HERE ////////////////////////////////////////////
            // SIMILAR TO PREVIOUS, WHEN WE DRAW RECTANGLES,
            // Actually if we convert everything into circle, we can remove many code in
            //      this file. But the backend needs to be modified,
            //      especially combine starter and middle from line 875.


            // first lets deal with alignment view
            d3.selectAll('.start_cell')
                .style('stroke', 'none');
            d3.selectAll('.middle_cell')
                .style('stroke', 'none');
            d3.select(this)
                .style('stroke', 'black')
                .style('stroke-width', '2px');
            
            // Get the Y coordinate of the selected circle
            var selectedCircleY = this.transform.baseVal.consolidate()['matrix']['f'];
            // Scroll to the selected circle
            document.getElementById("alignment_view_body").scrollTop= selectedCircleY;

            // first remove innerHTML of detail_view
            document.getElementById('detail_view_body').innerText = "";
            document.getElementById('arrow').innerText = "";
            // remove other arrow_links
            d3.selectAll('.arrow_link').remove();


            //////////////////////// add some arrows to it ///////////////////////////
            let curr_slider_value = document.getElementById('myRange').value;
            var arrowlineGenerator = d3.line()
                .curve(d3.curveNatural);
            var arrow_points = shift_arrow_path[k][i + 1];

            if (shifted === false) {
                arrow_points = unshift_arrow_path[k][i + 1];
            }

            arrow_order = [];

            for (var ct = 0; ct < arrow_points.length; ct++) {
                curr_arrow_nb = k;
                curr_arrow_cell = i + 1;
                let the_obj = arrow_points[ct];
                if (the_obj['count'] >= curr_slider_value) {
                    // do it
                    arrow_order.push(ct);
                    d3.select('.arrow_layer')
                        .append('path')
                        .attr('class', 'arrow_link')
                        .attr('d', arrowlineGenerator(the_obj['position'][0]));
                }
            }
            //////////////////// end of adding arrows ////////////////////////////////

            // {# then append something #}
            var detail_div = d3.select('#detail_view_body');
            var number_of_cells = details_data[k].length;

            var detail_cell_width = 460;
            var detail_cell_height = 100;
            var detail_container_height = 113;


            // TAG preparation ////////////////////////////////////////////////////////////////////
            // 1. found the variable name corpus
            const var_name_corpus = all_variable_names[k][true_index];


            // 2. found specific variables
            //    notice that, the above corpus contains the below variables
            //    Also, we need to record linked cells
            var linked_cells = [];
            var linked_cell_vars = [];
            const curr_cell_info = unshift_arrow_path[k][i + 1];
            var used_name_arr = [];
            if (curr_cell_info !== undefined) {
                for (var count = 0; count < curr_cell_info.length; count++) {
                    // if line was drawn, then I will consider its variables
                    //    as 'used'
                    if (curr_cell_info[count]['count'] >= curr_slider_value) {
                        used_name_arr = used_name_arr.concat(curr_cell_info[count]['var_names']);
                        linked_cells.push(curr_cell_info[count]['index']);
                        linked_cell_vars.push(curr_cell_info[count]['var_names']);
                    }
                }
                const unique_set = new Set(used_name_arr);
                used_name_arr = [...unique_set];
            }

            // 3. current searched_cell variables:
            const curr_searched_cell_var = store_searched_cell_vars[k];
            // end of TAG preparation ///////////////////////////////////////////////////////////////


            for (let j = 0; j < number_of_cells; j++) {
                var a_container = detail_div.append('div')
                    .attr('class', 'the_containers')
                    .attr('id', 'container' + j.toString())
                    .style('top', (j * (5)).toString() + 'px');


                // {# change the style based on ratio. If ratio = 1, make it different than others #}
                if (emb_ratio[k][j] === 1) {
                    a_container.style('border', '2px solid')
                        .style('border-color', color[k]);
                } else {
                    a_container.style('border', '1px solid');
                }

                var a_bar = a_container.append('div')
                    .attr('class', 'progress_bar')
                    .attr('id', 'the_bar' + j.toString())
                    .style('width', (detail_cell_width * emb_ratio[k][j]).toString() + 'px')
                    // {#.style('height', '10px')#}
                    .style('background-color', color[k]);
                // {#.style('display', 'block');#}



                // TAGGGGGGGGGGGGG
                if (true_index === j) {
                    // OK this is the one I'm looking for

                    // if nothing, then it's meaningless to do that.
                    if (var_name_corpus !== undefined) {
                        if (var_name_corpus.length > 0) {
                            var the_tag_div = a_container.append('div')
                                .attr('class', 'tag_container')
                                .attr('id', 'the_tag_div' + j.toString());
                            // change the size of biggest container
                            // a_container.style('height', (store_detail_height[k][j] + 43).toString() + 'px');

                            for (var loop = 0; loop < var_name_corpus.length; loop++) {
                                const the_name = var_name_corpus[loop];
                                const if_includes = used_name_arr.includes(the_name);
                                const if_searched_cell_var = curr_searched_cell_var.includes(the_name);

                                the_tag_div.append('div')
                                    .attr('class', 'word_tag')
                                    .attr('id', 'tag' + loop.toString())
                                    .style('left', ((loop + 1) * 6).toString() + 'px')
                                    .text(the_name)
                                    .style('background-color', function(d, i) {
                                        if (if_includes) {
                                            // return 'rgb(238, 210, 238)';
                                            return '#c7c7c7';
                                        }
                                        return 'white';
                                    })
                                    .style('border', function(d, i) {
                                        if (if_searched_cell_var) {
                                            return '2px solid';
                                        }
                                    });
                            }
                        }
                    }
                } else if (linked_cells.includes(j)) {
                    // linked cells
                    var a_tag_div = a_container.append('div')
                        .attr('class', 'tag_container')
                        .attr('id', 'the_tag_div' + j.toString());
                    // change the size of biggest container

                    const curr_var_name_corpus = all_variable_names[k][j];


                    for (var looop = 0; looop < curr_var_name_corpus.length; looop++) {
                        const the_name = curr_var_name_corpus[looop];
                        // find correct position
                        const posi = linked_cells.findIndex((element) => element === j);
                        const if_inc = linked_cell_vars[posi].includes(the_name);
                        const if_search_cell_var = curr_searched_cell_var.includes(the_name);

                        a_tag_div.append('div')
                            .attr('class', 'word_tag')
                            .attr('id', 'tag' + looop.toString())
                            .style('left', ((looop + 1) * 6).toString() + 'px')
                            .text(the_name)
                            .style('background-color', function(d, i) {
                                if (if_inc) {
                                    // return 'rgb(238, 210, 238)';
                                    return '#c7c7c7';
                                }
                                return 'white';
                            })
                            .style('border', function(d, i) {
                                if (if_search_cell_var) {
                                    return '2px solid';
                                }
                            });
                    }

                }

                // end of TAGGGGGGGGGGG

                var a_detail_text = a_container.append('div')
                    .attr('class', 'fake_textarea detail_view_text')
                    .attr('id', 'detail_view_text' + j.toString())
                    .attr('disabled', 'true')
                    .append('pre')
                    .append('code')
                    .attr('id', 'detail_fake_code' + j.toString())
                    .attr('class', 'python hljs');

                document.getElementById('detail_fake_code' + j.toString()).innerHTML =
                    hljs.highlight('python', details_data[k][j].conc_cell, 'true').value;

            }

            // // {# we scroll to that position, then make it different for a while #}
            var myElement = document.getElementById('container' + (true_index).toString());
            document.getElementById('detail_view').scrollTop = myElement.offsetTop;


            // scroll to proper position of list_view
            var myElement2 = document.getElementById('list_cell' + k.toString());
            document.getElementById('list_view').scrollTop = myElement2.offsetTop;

            d3.select('#arrow')
                .append('p')
                .attr('id', 'detail_left_arrow')
                .style('position', 'absolute')
                // .style('left', (detail_cell_width + 20).toString() + 'px')
                .style('top', myElement.offsetTop.toString() + 'px');
            document.getElementById('detail_left_arrow').innerText = '←';
        })
        .on('mouseover', handleMOver_circle)
        .on('mouseout', function(d, i) {
            if (emb_ratio[k][real_index[k][i + 1]] === 1) {
                d3.select(this).attr("d", d3.symbol().type(d3.symbolCircle).size(120));
            } else {
                d3.select(this).attr("d", d3.symbol().type(d3.symbolCircle).size(50));
            }
        });

}


document.getElementsByTagName('svg')[0].style.height = (document.getElementsByTagName('g')[0].getBoundingClientRect().height + 10) + 'px';
document.getElementsByTagName('svg')[0].style.width = (document.getElementsByTagName('g')[0].getBoundingClientRect().width + 100) + 'px';

d3.select('.line_layer')
    .style('display', 'none');

function shift_alignment() {
    shifted = true;
    // {#  deal with paths  #}
    var areaGenerator = d3.area();
    areaGenerator.x0(function (d) {
        return d.left
    }).x1(function (d) {
        return d.right
    }).y(function (d) {
        return d.y
    });

    areaGenerator
        .defined(function (d) {
            return d !== null;
        });
    d3.select('.line_layer')
        .selectAll('path')
        .transition()
        .attr('d', function (d, i) {
            return areaGenerator(align_path_data[i]);
        });


    // deal with arrows
    d3.select('.arrow_layer')
        .selectAll('.arrow_link')
        .transition()
        .attr('d', function (d, i) {
            let arrowlineGenerator = d3.line()
                .curve(d3.curveNatural);
            let arrow_points = shift_arrow_path[curr_arrow_nb][curr_arrow_cell];
            return arrowlineGenerator(arrow_points[arrow_order[i]]['position'][0]);
        });

    // {# deal with starters #}
    d3.select('.cell_layer')
        .selectAll('.start_cell')
        .transition()
        .attr('transform', function (d, i) {
            return 'translate(' + align_start_pts[i] + ')';

        });

    // {# deal with middles #}
    d3.select('.cell_layer')
        .selectAll('.middle_cell')
        .transition()
        .attr('transform', function (d, i) {
            return 'translate(' + one_d_align_mid[i] + ')';
        });

    d3.select('.line_layer')
        .style('display', 'none');
}


function unshift_alignment() {
    shifted = false;
    // {#  deal with paths  #}
    var areaGenerator = d3.area();
    areaGenerator.x0(function (d) {
        return d.left
    }).x1(function (d) {
        return d.right
    }).y(function (d) {
        return d.y
    });

    areaGenerator
        .defined(function (d) {
            return d !== null;
        });
    d3.select('.line_layer')
        .selectAll('path')
        .transition()
        .attr('d', function (d, i) {
            return areaGenerator(unalign_path_data[i]);
        });

    // deal with arrows
    d3.select('.arrow_layer')
        .selectAll('.arrow_link')
        .transition()
        .attr('d', function (d, i) {
            console.log(i);
            let arrowlineGenerator = d3.line()
                .curve(d3.curveNatural);
            let arrow_points = unshift_arrow_path[curr_arrow_nb][curr_arrow_cell];
            return arrowlineGenerator(arrow_points[arrow_order[i]]['position'][0]);
        });

    // {# deal with starters #}
    d3.select('.cell_layer')
        .selectAll('.start_cell')
        .transition()
        .attr('transform', function (d, i) {
            return 'translate(' + unalign_start_pts[i] + ')';

        });

    // {# deal with middles #}
    d3.select('.cell_layer')
        .selectAll('.middle_cell')
        .transition()
        .attr('transform', function (d, i) {
            return 'translate(' + one_d_unalign_mid[i] + ')';
        });

    d3.select('.line_layer')
        .style('display', 'block');
}


function filter_arrow() {
    if (curr_selected_cell !== null) {
        curr_selected_cell.dispatch('click');
    }
}

// triger line view in alignment_view
unshift_alignment();


for (let h = 0; h < align_markdown_path_data.length; h++) {
    for (let m = 0; m < align_markdown_path_data[h].length; m++) {
        var one_path = areaGenerator(align_markdown_path_data[h][m]);

        d3.select('.line_layer')
            .append('path')
            .attr('d', one_path)
            .style('fill', function (d, i) {
                // return color[h];
                return 'white';
            }).style('opacity', 1);
    }
}
