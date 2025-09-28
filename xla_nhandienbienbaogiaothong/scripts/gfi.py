import sys
import yaml

def read_yaml(file_path):
    """Đọc file YAML và trả về dữ liệu dạng dict"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

if __name__ == "__main__":
    # Nếu có tham số dòng lệnh thì lấy file đó, không thì mặc định là data.yaml
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "xla_nhandienbienbaogiaothong/configs/data.yaml"
    
    config = read_yaml(yaml_file)
    print("Nội dung YAML:")
    print(config)
