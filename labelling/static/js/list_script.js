document.addEventListener('DOMContentLoaded', function() {
  // 파일 목록을 API에서 가져오기
  fetch('/api/files/')
      .then(response => response.json())
      .then(data => {
          const fileListUl = document.getElementById('file-list-ul');
          if (data.files && data.files.length > 0) {
              data.files.forEach(fileName => {
                  const listItem = document.createElement('li');
                  listItem.innerHTML = `<a href="/result/${fileName}/">${fileName}</a>`;
                  fileListUl.appendChild(listItem);
              });
          } else {
              fileListUl.innerHTML = '<li>저장된 파일이 없습니다.</li>';
          }
      })
      .catch(error => {
          document.getElementById('file-list').innerText = '파일 목록을 불러오는 중 오류가 발생했습니다.';
      });
});
