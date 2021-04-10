// Helper function which finds how many lines are there
//    in a cell
function find_line_count(a_str) {
    return (a_str.match(/\n/g) || []).length;
}

// Function for doing summation...
function partial_sum(arr, until_index) {
    var curr_sum = 0;
    for (let i = 0; i <= until_index; i++) {
        curr_sum = curr_sum + arr[i];
    }
    return curr_sum;
}
