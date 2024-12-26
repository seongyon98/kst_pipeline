document.addEventListener('DOMContentLoaded', function() {
  const fileName = "{{ file_name }}";  // 템플릿에서 파일 이름을 전달받습니다.
  fetch(`/api/results/${fileName}/`)
      .then(response => response.json())
      .then(data => {
          const resultDiv = document.getElementById('result');
          if (data) {
              resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
          } else {
              resultDiv.innerHTML = '<p>결과를 불러오는 중 오류가 발생했습니다.</p>';
          }
      })
      .catch(error => {
          document.getElementById('result').innerText = '결과를 불러오는 중 오류가 발생했습니다.';
      });
});
