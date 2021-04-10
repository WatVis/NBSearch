// This file contains necessary preprocessing steps.


// we need to convert middle points into 1-d array
//             we do this for later transition part
one_d_unalign_mid = [];
for (let i = 0; i < unalign_middle_pts.length; i++) {
    for (let j = 0; j < unalign_middle_pts[i].length; j++) {
        one_d_unalign_mid.push(unalign_middle_pts[i][j]);
    }
}
console.log(one_d_unalign_mid);

// same as above, but for Dots_view
one_d_align_mid = [];
for (let i = 0; i < align_middle_pts.length; i++) {
    for (let j = 0; j < align_middle_pts[i].length; j++) {
        one_d_align_mid.push(align_middle_pts[i][j]);
    }
}
console.log(one_d_align_mid);

// convert embeddings into ratio
emb_ratio = [];
for (let i = 0; i < emb_data.length; i++) {
    var curr_max = Math.max(...emb_data[i]);
    var one_notebook = [];
    for (let j = 0; j < emb_data[i].length; j++) {
        if (curr_max === 0) {
            one_notebook.push(0);
        } else {
            one_notebook.push(emb_data[i][j] / curr_max);
        }

    }
    emb_ratio.push(one_notebook);
}
console.log(emb_ratio);

// deal with length, the detail_view length
var store_detail_height = [];
for (let i = 0; i < details_data.length; i++) {
    var curr_height = [];
    for (let j = 0; j < details_data[i].length; j++) {
        curr_height.push((find_line_count(details_data[i][j].conc_cell) + 1) * 20);
    }
    store_detail_height.push(curr_height);
}
console.log(store_detail_height);


// deal with result height
var store_result_height = [];
for (let i = 0; i < fc.length; i++) {
    store_result_height.push((find_line_count(fc[i].conc_cell) + 1) * 20);
}




// find searched_cells' variables
var store_searched_cell_vars = [];
for (var i = 0; i < emb_ratio.length; i++) {
    for (var j = 0; j < emb_ratio[i].length; j++) {
        if (emb_ratio[i][j] === 1) {
            store_searched_cell_vars.push(all_variable_names[i][j]);
            break;
        }
    }
}
console.log(store_searched_cell_vars);



// for demonstrating previously selected parameters
document.getElementById('option_10').selected = false;
if (returned_nbs === 5) {
    document.getElementById('option_5').selected = true;
} else if (returned_nbs === 10) {
    document.getElementById('option_10').selected = true;
} else if (returned_nbs === 15) {
    document.getElementById('option_15').selected = true;
} else {
    document.getElementById('option_20').selected = true;
}

// for demonstrating previously selected parameters
document.getElementById('option50').selected = false;
if (mkd_prev_value === 0) {
    document.getElementById('option0').selected = true;
} else if (mkd_prev_value === 10) {
    document.getElementById('option10').selected = true;
} else if (mkd_prev_value === 20) {
    document.getElementById('option20').selected = true;
} else if (mkd_prev_value === 30) {
    document.getElementById('option30').selected = true;
} else if (mkd_prev_value === 40) {
    document.getElementById('option40').selected = true;
} else if (mkd_prev_value === 50) {
    document.getElementById('option50').selected = true;
} else if (mkd_prev_value === 60) {
    document.getElementById('option60').selected = true;
} else if (mkd_prev_value === 70) {
    document.getElementById('option70').selected = true;
} else if (mkd_prev_value === 80) {
    document.getElementById('option80').selected = true;
} else {
    document.getElementById('option90').selected = true;
}
