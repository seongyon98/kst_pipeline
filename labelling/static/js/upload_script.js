function getCSRFToken() {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'csrftoken') return value;
    }
    return '';
}

// 선택된 파일들을 목록에 추가하는 함수
function updateFileList() {
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');

    // 기존 파일 목록을 초기화하지 않고 유지
    const selectedFiles = Array.from(fileInput.files);

    // 기존 파일 목록을 비우고 새로운 파일 목록을 추가
    fileList.innerHTML = '';
    selectedFiles.forEach((file, index) => {
        const li = document.createElement('li');
        li.textContent = file.name;

        // 삭제 버튼 생성
        const deleteButton = document.createElement('button');
        deleteButton.textContent = '삭제';
        deleteButton.classList.add('remove-text');  // 스타일을 위한 클래스 추가
        deleteButton.onclick = () => removeFile(index);

        li.appendChild(deleteButton);
        fileList.appendChild(li);
    });
}

// 파일 삭제 함수
function removeFile(index) {
    const fileInput = document.getElementById('file-input');
    const selectedFiles = Array.from(fileInput.files);

    // 선택된 파일들을 갱신
    selectedFiles.splice(index, 1);

    // file-input에 새로운 파일 목록 반영
    const dataTransfer = new DataTransfer();
    selectedFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;

    // 파일 목록 갱신
    updateFileList();
}

// 업로드 기능
function uploadFiles() {
    const formData = new FormData();
    const files = document.getElementById('file-input').files;
    if (files.length === 0) {
        alert("파일을 선택해주세요.");
        return;
    }

    // 선택한 모든 파일을 formData에 추가
    for (let i = 0; i < files.length; i++) {
        formData.append("file", files[i]);
    }

    // CSRF 토큰 추가
    const csrfToken = getCSRFToken();

    // 프로그래스바를 보이도록 설정
    document.getElementById('progress-container').style.display = 'block';

    // Fetch API를 사용하여 파일 업로드
    fetch("/api/upload/", {
        method: "POST",
        headers: {
            'X-CSRFToken': csrfToken,
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            document.getElementById('upload-status').innerText = data.message;
            const fileNames = data.file_names;  // 업로드된 파일 이름들을 저장
            document.getElementById('uploaded-file-name').innerText = `업로드한 파일들: ${fileNames.join(', ')}`;
            document.getElementById('uploaded-file-name').style.display = 'block';
        } else {
            document.getElementById('upload-status').innerText = data.error;
        }
        document.getElementById('progress-container').style.display = 'none';
    })
    .catch(error => {
        document.getElementById('upload-status').innerText = '업로드 중 오류가 발생했습니다.';
        document.getElementById('progress-container').style.display = 'none';
    });
}


function uploadText() {
    const textInputs = document.querySelectorAll('.text-input');
    if (textInputs.length === 0) {
        alert("텍스트를 입력해주세요.");
        return;
    }

    const formData = new FormData();

    // 여러 텍스트 입력 필드를 formData에 추가
    textInputs.forEach((input, index) => {
        if (input.value.trim() !== "") {
            formData.append("text", input.value);
        }
    });

    // CSRF 토큰 추가
    const csrfToken = getCSRFToken();

    // 프로그래스바를 보이도록 설정
    document.getElementById('progress-container').style.display = 'block';

    // Fetch API를 사용하여 텍스트 업로드
    fetch("/api/upload/", {
        method: "POST",
        headers: {
            'X-CSRFToken': csrfToken,
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            document.getElementById('upload-status').innerText = data.message;
            const fileNames = data.file_names;  // 업로드된 텍스트의 이름들을 저장
            document.getElementById('uploaded-file-name').innerText = `업로드한 텍스트들: ${fileNames.join(', ')}`;
            document.getElementById('uploaded-file-name').style.display = 'block';
            document.getElementById('show-result-btn').style.display = 'inline-block'; // 결과 확인 버튼 보이기
        } else {
            document.getElementById('upload-status').innerText = data.error;
        }
        document.getElementById('progress-container').style.display = 'none';
    })
    .catch(error => {
        document.getElementById('upload-status').innerText = '업로드 중 오류가 발생했습니다.';
        document.getElementById('progress-container').style.display = 'none';
    });
}

// 텍스트 입력 필드 추가
function addTextInput() {
    const textInputContainer = document.getElementById('text-inputs');
    const newTextInputWrapper = document.createElement('div');
    newTextInputWrapper.classList.add('text-input-wrapper');

    const newTextInput = document.createElement('textarea');
    newTextInput.classList.add('text-input');
    newTextInput.rows = 6;
    newTextInput.placeholder = "여기에 문제를 입력하세요...";

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.classList.add('remove-text');
    removeButton.innerText = '삭제';
    removeButton.style.display = 'inline-block';
    removeButton.onclick = () => removeTextInput(removeButton);

    newTextInputWrapper.appendChild(newTextInput);
    newTextInputWrapper.appendChild(removeButton);
    textInputContainer.appendChild(newTextInputWrapper);
}

// 텍스트 입력 필드 삭제
function removeTextInput(button) {
    const textInputWrapper = button.parentElement;
    textInputWrapper.remove();
}


function fetchAllResults() {
    const csrfToken = getCSRFToken();
    const uploadedFileNames = document.getElementById('uploaded-file-name').innerText.split(': ')[1].split(', ');

    fetch("/api/all-results/", {
        method: "POST",
        headers: {
            'X-CSRFToken': csrfToken,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_names: uploadedFileNames })
    })
    .then(response => response.json())
    .then(data => {
        const resultDisplay = document.getElementById('results');
        resultDisplay.innerHTML = '';
        if (data && data.length > 0) {
            data.forEach(file => {
                resultDisplay.innerHTML += `
                    <h3>파일: ${file.file_name}</h3>
                    <p>상태: ${file.status}</p>
                    <p>라벨링 결과: ${file.labeling_result || '처리 중'}</p>
                    <hr>
                `;
            });
        } else {
            resultDisplay.innerHTML = '<p>결과를 찾을 수 없습니다.</p>';
        }
    })
    .catch(error => {
        console.error('결과 조회 중 오류가 발생했습니다:', error);
    });
}
