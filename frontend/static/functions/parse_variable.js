var something = python.parse('plt.plot(ST,Payoff_LongSpread)\n' +
    'plt.show()\n');
console.log(something);
console.log(something.length);
console.log(something.toString());

for (let i = 0; i < fc.length; i++) {
    console.log(i);
    console.log(fc[i].conc_cell);
    let ans = python.parse(fc[i].conc_cell + '\n');
    console.log(ans);



}