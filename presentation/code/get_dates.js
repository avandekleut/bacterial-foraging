let dates_list = document.getElementsByClassName('gs_a');
let dates = [];
let date_re = new RegExp('20[0-9]{2}')
for (let date of dates_list){
    dates.push(parseInt(date.innerText.match(date_re)[0]));
}

let citations_list = document.getElementsByClassName('gs_fl');
let citations = [];
let citations_re = new RegExp('Cited by [0-9]{+}')
for (let citation of citations_list){
    let match = citation.innerText.match(new RegExp('Cited by [0-9]+'));
    if (match){
        citations.push(parseInt(match[0].match(new RegExp('[0-9]+'))[0]));
    }
}

console.log(dates)
console.log(citations)
