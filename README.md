
# CS331 - Advanced Computer Vision: Final Project

## Project Overview
This repository contains the final project for the **CS331 - Advanced Computer Vision** course. The project was completed as part of the requirements for the course and focuses on evaluating Vision-Language Models (VLMs) and implementing specific methodologies related to the field of computer vision.

---

## Team Members
| No. | Name                   | Student ID |
| --- | ---------------------- | ---------- |
| 1   | Đặng Hữu Phát         | 22521065   |
| 2   | Phan Hoàng Phước      | 22521156   |
| 3   | Nguyễn Hữu Hoàng Long | 22520817  |

---

## Acknowledgements
Our group sincerely thanks **PhD. Mai Tiến Dũng**, Lecturer of the Department of Computer Science at the University of Information Technology, Vietnam National University - Ho Chi Minh City, for his dedicated teaching and invaluable feedback on our project.

Throughout the completion of this project, we made significant efforts to ensure the best possible outcome. However, despite our diligence, there were some unintended mistakes. We kindly request honest feedback and constructive suggestions from the professor to further enhance our work. 

For any inquiries or feedback, feel free to contact us via email.

---

## Instructions to Run the Code

### 1. Environment Setup
We recommend creating a new Conda environment with Python 3.12:
```bash
conda create -n vision_env python=3.12
conda activate vision_env
```

Install CUDA and PyTorch:
```bash
conda install cuda -c nvidia/label/cuda-12.1.0
conda install pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Clone the Project
Clone this repository to your local machine:
```bash
git clone https://github.com/HuuPhat125/CS331---Advanced-Computer-Vision.git
cd CS331---Advanced-Computer-Vision
```

### 3. Install Required Libraries
Install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Evaluation

### Evaluating Vision-Language Models
Navigate to the `evaluation` directory:
```bash
cd evaluation
```

Run the script to generate responses:
```bash
python generate_response.py --output_dir <output_directory> --output_file <output_file_name> --model <model_name> --batch_size <batch_size>
```

The results will be saved as a JSON file in the specified `output_dir`. Example structure of the JSON output:
```json
{
    "0": {
        "questionId": "test_1",
        "question": "Bộ dụng cụ như hình vẽ có thể dùng để thu khí nào sau đây ? <image_1>",
        "options": "['A. H2.', 'B. NH3.', 'C. CH4.', 'D. Cl2.']",
        "answer": "D",
        "explanation": "Đáp án đúng là: DVì ống nghiệm ngửa nên khí thu được nặng hơn không khí→ chọn Cl2",
        "subject": "Hóa học",
        "grade": "Lớp 10",
        "image": "test_1.png",
        "caption": "",
        "ocr": [],
        "query": "Câu hỏi: Bộ dụng cụ như hình vẽ có thể dùng để thu khí nào sau đây ? <image_1>\nCác lựa chọn:\nA. H2.\nB. NH3.\nC. CH4.\nD. Cl2.\nDựa vào nội dung câu hỏi và nội dung trong bức ảnh hãy chọn đáp án đúng và giải thích bằng Tiếng Việt",
        "response": "Đáp án đúng là **B. NH3**.\n\n**Giải thích:**\n\nBộ dụng cụ như hình vẽ là phương pháp thu khí bằng cách đẩy không khí. Ống nghiệm được úp ngược xuống, chứng tỏ khí thu được nhẹ hơn không khí. Trong các khí đã cho, chỉ có NH3 là khí nhẹ hơn không khí.\n\nCác khí còn lại:\n\n* H2: Nhẹ hơn không khí nhưng phải thu bằng cách đẩy nước vì nó tan trong nước.\n* CH4: Nhẹ hơn không khí nhưng phải thu bằng cách đẩy nước vì nó tan trong nước.\n* Cl2: Nặng hơn không khí nên không thể thu bằng cách đẩy không khí.\n\nVậy đáp án đúng là B. NH3."
    }
}
```

After generating the responses, extract the correct answers (A, B, C, D) from the model's outputs:
```bash
python extract_answer.py
```

---

## Our method
The training and evaluation code for our method is located in the `train_model` directory. To run:
```bash
cd train_model
```

Run the script to train model:
```bash
python run.py --image_model <image_model> --is_ocr <is_ocr> --learning_rate <learning_rate> --num_epochs <num_epochs>
```

```image_model```: can use in ```["resnet, "resnext", "vit"]```.
```is_ocr```: boolean value.

The result will get a model file name ```model.pkl```. To predict, using ```predict``` function in the ```run.py``` file.
