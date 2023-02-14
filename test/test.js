const axios = require('axios');

const data = {
  prompt: 'Vietnam là nước nào ?',
  length: 50,
  temperature: 0.5,
};

axios.post('http://127.0.0.1:5000/api/complete', data)
  .then((response) => {
    console.log(response.data.output);
  })
  .catch((error) => {
    console.error(error);
  });