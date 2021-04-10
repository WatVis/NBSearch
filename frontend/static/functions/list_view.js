var curr_selected_cell = null;

///////////////// This file contains rendering of list_view, and some necessary interactions are being set up here. //////////

// d3 select the canvas where we will draw
var list_div = d3.select('#list_view_body');

// Firstly fill in the list_view with fake code cells
for (let i = 0; i < fc.length; i++) {
    var a_list_cell_container = list_div.append('div')
        .attr('class', 'list_cell')
        .attr('id', 'list_cell' + i.toString())
        .style('top', (i * (5)).toString() + 'px');

    var the_text = a_list_cell_container.append('div')
        .attr('id', 'list_view_text' + i.toString())
        .attr('class', 'fake_textarea list_view_text')
        .attr('disabled', 'true')
        .style('border-color', color[i])
        .append('pre')
        .append('code')   // wrap code with 'pre' and 'code', required by highlight.js
        .attr('id', 'list_fake_code' + i.toString())
        .attr('class', 'python hljs'); // By setting class as 'python hljs', highlight.js
                                       // will automatically convert plain-text code into
                                       // colorful texts which are wrapped by different
                                       // components
    document.getElementById('list_fake_code' + i.toString()).innerText = fc[i].conc_cell;
}


// Select all cells, set their onClick reaction.
var add_detail_view_cells = list_div.selectAll('.list_cell')
    .on('click', function (d, i) {
        // i is the index, e.g., the second cell will automatically have i == 1.
        curr_selected_cell = null;
        console.log('clicked!');
        //  first change alignment_view
        //  Remove potentially existed node click effects from alignment_view
        d3.selectAll('.start_cell')
            .style('stroke', 'none');
        d3.selectAll('.middle_cell')
            .style('stroke', 'none');

        //  first remove innerHTML of detail_view
        document.getElementById('detail_view_body').innerText = "";
        // remove other arrow_links
        d3.selectAll('.arrow_link').remove();
        // then start append something onto detail_view
        var detail_div = d3.select('#detail_view_body');
        var number_of_cells = details_data[i].length;

        var detail_cell_width = 460;
        var detail_cell_height = 100;

        for (let j = 0; j < number_of_cells; j++) {
            // detail_div is defined lines above,
            // Continuously call '.append()' in a for loop
            // will keep adding components to it as it's CHILDREN.
            var a_container = detail_div.append('div')
                .attr('class', 'the_containers')
                .style('top', (j * (5)).toString() + 'px');

            // change the style based on ratio. If ratio = 1, make it different than others
            if (emb_ratio[i][j] === 1) {
                a_container.style('border', '2px solid')
                    .style('border-color', color[i]);

            } else {
                a_container.style('border', '1px solid');
            }

            // Progress bar appears before code area.
            var a_bar = a_container.append('div')
                .attr('class', 'progress_bar')
                .attr('id', 'the_bar' + j.toString())
                .style('width', (detail_cell_width * emb_ratio[i][j]).toString() + 'px')
                .style('background-color', color[i]);
            // code area
            var a_detail_text = a_container.append('div')
                .attr('class', 'fake_textarea detail_view_text')
                .attr('id', 'detail_view_text' + j.toString())
                .attr('disabled', 'true')
                .append('pre')
                .append('code')
                .attr('id', 'detail_fake_code' + j.toString())
                .attr('class', 'python hljs');
            document.getElementById('detail_fake_code' + j.toString()).innerHTML =
                hljs.highlight('python', details_data[i][j].conc_cell, 'true').value;


        }

        // Then, since user clicked a cell from list_view, we should
        //        indicate the correpsonding cell in detail_view and scroll to
        //        a proper position.

        // find which one of emb_ratio == 1
        for (let ct = 0; ct < emb_ratio[i].length; ct++) {
            if (emb_ratio[i][ct] === 1) {
                // trigger click
                curr_selected_cell = d3.select('#cell' + i.toString() + (real_index[i].indexOf(ct)).toString());
                curr_selected_cell.dispatch('click');
                //scroll to proper alignment_view position
                if (ct === 0) {
                    document.getElementById('alignment_view').scrollTop = unalign_start_pts[i][real_index[i].indexOf(ct)] - 30;
                } else {
                    document.getElementById('alignment_view').scrollTop = unalign_middle_pts[i][real_index[i].indexOf(ct) - 1][1] - 30;
                }
                break;
            }
        }
    });
