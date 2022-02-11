function showCustomInput(that) {
            if (that.value == "custom") {
                document.getElementById("custom_weights_div").style.display = "block";
                document.getElementById("ticker_div").style = "float:left;width:20%;";
                document.getElementById("strategy_div").style = "float:left;width:20%;";
                document.getElementById("history_div").style = "float:left;width:20%;";
                document.getElementById("wealth_div").style = "float:left;width:20%;";
            } else {
                document.getElementById("custom_weights_div").style.display = "none";
                document.getElementById("ticker_div").style = "float:left;width:25%;";
                document.getElementById("strategy_div").style = "float:left;width:25%;";
                document.getElementById("history_div").style = "float:left;width:25%;";
                document.getElementById("wealth_div").style = "float:left;width:25%;";
            }
        }


function updateTC(range) {
    console.log(range.value);
    const net_data = net[rb.value][tc.value];
    chart.data.datasets[1].data = net_data;
    chart.update();
}

function updateRF(rebalancing_frequency) {
    console.log(rebalancing_frequency.value);
    const gross_data = gross[rb.value];
    const net_data = net[rb.value][tc.value];
    chart.data.datasets[0].data = gross_data;
    chart.data.datasets[1].data = net_data;
    chart.update();
}
