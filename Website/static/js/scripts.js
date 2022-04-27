var pieVl = {
  $schema: 'https://vega.github.io/schema/vega-lite/v4.json',
  description: 'A simple bar chart with embedded data.',
  data: {
    values: [
      { name: 'test', featureImportance: 0 }
    ]
  },
  mark: {
    type: 'arc',
    tooltip: true
  },
  selection: {
    'feature': {
      type: 'multi',
      fields: ['name'],
      bind: "legend"
    }
  },
  encoding: {
    color: {
      condition: {
        legend: { title: "Feature Name" },
        selection: 'feature',
        field: 'name',
        type: 'nominal',
        scale: {
          scheme: 'tableau20'
        }
      },
      value: "grey"
    },
    theta: {
      field: 'featureImportance',
      type: 'quantitative'
    }
  },
  view: {
    stroke: null
  },
  autosize: {
    type: 'fit',
    contains: 'padding'
  },
  width: "container",
  height: 250
};

var lineVl = {
  $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
  description: 'Stock prices of 5 Tech Companies over Time.',
  data: {
    values: [
      { name: 'test', featureImportance: 0 }
    ]
  },
  mark: {
    type: 'line',
    point: { size: 100 },
    tooltip: true
  },
  selection: {
    'feature': {
      type: 'multi',
      fields: ['name'],
      bind: "legend"
    }
  },
  encoding: {
    x: {
      title: 'All Inputs',
      field: 'session',
      type: 'quantitative',
      axis: { tickMinStep: 1 }
    },
    y: {
      title: 'Feature Importance',
      field: 'featureImportance',
      type: 'quantitative',
      scale: { zero: false },
    },
    color: {
      condition: {
        selection: 'feature',
        field: 'name',
        type: 'nominal',
        scale: {
          scheme: 'tableau20'
        },
        legend: { title: "Feature Name" }
      },
      value: 'grey',
      legend: { title: "Feature Name" }
    },
    opacity: {
      condition: {
        selection: 'feature',
        value: 1
      },
      value: 0.2
    },
    tooltip: [
      { field: 'name', type: 'nominal' },
      { field: 'featureImportance', type: 'quantitative' }
    ]
  },
  autosize: {
    type: 'fit',
    contains: 'padding'
  },
  width: "container",
  height: 300
}

var count = 0;

function parseForm() {
  const button = document.getElementById("submit-button");
  button.classList.add('is-loading')
  const data = {
    sex: 0,
    age: 25,
    race: 0,
    juv_fel_count: 0,
    juv_misd_count: 0,
    juv_other_count: 0,
    priors_count: 0,
    is_recid: 0,
    is_violent_recid: 0,
    c_charge_degree_F: 0,
    c_charge_degree_M: 0,
    r_charge_degree: "",
    threshold: 0.0,
    pareto_index: 0,
    session: count
  }

  const male = document.getElementById("sex_male");
  const female = document.getElementById("sex_female");
  data.sex = male.checked ? male.value : female.value;

  const age = document.getElementById("age");
  data.age = age.value;

  const caucasian = document.getElementById("race_c");
  const africanAmerican = document.getElementById("race_aa");
  data.race = caucasian.checked ? caucasian.value : africanAmerican.value;

  const juv_fel_count = document.getElementById("juv_fel_count");
  data.juv_fel_count = juv_fel_count.value;

  const juv_misd_count = document.getElementById("juv_misd_count");
  data.juv_misd_count = juv_misd_count.value;

  const juv_other_count = document.getElementById("juv_other_count");
  data.juv_other_count = juv_other_count.value;

  const priors_count = document.getElementById("priors_count");
  data.priors_count = priors_count.value;

  const is_recid_y = document.getElementById("is_recid_y");
  const is_recid_n = document.getElementById("is_recid_n");
  data.is_recid = is_recid_y.checked ? is_recid_y.value : is_recid_n.value;

  const is_violent_recid_y = document.getElementById("is_violent_recid_y");
  const is_violent_recid_n = document.getElementById("is_violent_recid_n");
  data.is_violent_recid = is_violent_recid_y.checked ? is_violent_recid_y.value : is_violent_recid_n.value;

  const c_charge_degree_F = document.getElementById("c_charge_degree_F");
  data.c_charge_degree_F = c_charge_degree_F.checked ? 1 : 0

  const c_charge_degree_M = document.getElementById("c_charge_degree_M");
  data.c_charge_degree_M = c_charge_degree_M.checked ? 1 : 0

  const r_charge_degree = document.getElementById("r_charge_degree");
  data.r_charge_degree = r_charge_degree.value;


  const threshold = document.getElementById("fpfn");
  data.threshold = threshold.value;

  const pareto_index = document.getElementById("pareto_index");
  data.pareto_index = pareto_index.value;

  count++;
  scoreData(data).then(scoredData => {

    const prediction = document.getElementById("prediction");
    prediction.innerText = scoredData.prediction == 1 ? "Labeled will Reoffend" : "Labeled will Not Reoffend"

    const accuracy = document.getElementById("accuracy");
    percentageAcc = parseFloat(scoredData.accuracy).toFixed(2);
    accuracy.innerText = "Accuracy: " + percentageAcc + "%";
    const prevData = localStorage.getItem("previousFeatureImportance");

    var updatedData;
    if (prevData.length > 0) {
      updatedData = [...JSON.parse(prevData), scoredData.featureImportance];
      localStorage.setItem("previousFeatureImportance", JSON.stringify(updatedData));
      lineVl.data.values = JSON.stringify(updatedData.flat());
      vegaEmbed('#line_chart', lineVl);

    } else {
      localStorage.setItem("previousFeatureImportance", JSON.stringify([...prevData, scoredData.featureImportance]));
      lineVl.data.values = JSON.stringify(scoredData.featureImportance);
      vegaEmbed('#line_chart', lineVl);
    }

    pieVl.data.values = JSON.stringify(scoredData.featureImportance);
    vegaEmbed('#pie_chart', pieVl);

    drawFlowChart(scoredData.flowChart);

    button.classList.remove('is-loading')
  })
}

function parseThreshold() {

}


const drawFlowChart = (rows) => {
  if (rows) {
    var data = new google.visualization.DataTable();
    data.addColumn('string', 'From');
    data.addColumn('string', 'To');
    data.addColumn('number', 'Count');
    data.addRows(rows);

    // Sets chart options.
    var options = {
      width: 600,
    };

    // Instantiates and draws our chart, passing in some options.
    var chart = new google.visualization.Sankey(document.getElementById('flowChart'));
    chart.draw(data, options);
  }
}

const scoreData = async (data) => {
  const location = window.location.hostname;
  const settings = {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  };
  try {
    //Local build
    const fetchResponse = await fetch(`http://${location}:5000/predict`, settings);
    //Heroku build
    //const fetchResponse = await fetch(`https://${location}/predict`, settings);
    const responseJson = await fetchResponse.json();
    return responseJson;
  } catch (e) {
    return e;
  }

}

localStorage.setItem("previousFeatureImportance", []);

const submit = document.getElementById("submit-button");
submit.addEventListener("click", parseForm);

parseForm()

google.charts.load('current', { 'packages': ['sankey'] });
google.charts.setOnLoadCallback(drawFlowChart);




