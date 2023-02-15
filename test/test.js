const axios = require('axios');

const data = {
  prompt: 'We have done a lot before to improve the quality and ',
  length: 80,
  temperature: 0.1,
};

axios.post('http://127.0.0.1:5000/api/complete', data)
  .then((response) => {
    console.log(response.data.output);
  })
  .catch((error) => {
    console.error(error);
  });