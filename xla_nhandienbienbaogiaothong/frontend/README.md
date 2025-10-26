# Frontend - Nhận diện biển báo giao thông

Giao diện React với Vite và Tailwind CSS để phát hiện biển báo giao thông.

## 📋 Tính năng

- 📷 Upload ảnh và phát hiện biển báo
- 🎥 Upload video và xử lý real-time
- 📊 Hiển thị kết quả chi tiết với statistics
- 🎨 Giao diện đẹp, responsive với Tailwind CSS
- ⚡ Nhanh và mượt với Vite

## 🚀 Cài đặt

```bash
cd frontend
npm install
```

## 🎯 Chạy Development Server

```bash
npm run dev
```

Mở trình duyệt tại: `http://localhost:3000`

## 🔨 Build Production

```bash
npm run build
```

## 📦 Dependencies

- **React 18.3** - UI library
- **Vite 5.0** - Build tool
- **Tailwind CSS 3.4** - Styling
- **Axios** - HTTP client cho API calls

## 🎨 Components

### `App.jsx`
Main application component với tab switching giữa image và video detection.

### `UploadImage.jsx`
- Upload ảnh qua file input hoặc drag & drop
- Preview ảnh trước khi upload
- Loading state khi đang xử lý
- Error handling

### `VideoDetector.jsx`
- Upload video files
- Preview video
- Hiển thị cảnh báo về thời gian xử lý
- Download video đã xử lý

### `ResultDisplay.jsx`
- Hiển thị ảnh/video đã được annotate
- Statistics (tổng số biển báo, độ tin cậy trung bình)
- Chi tiết từng phát hiện với confidence score
- Color coding theo confidence level

## 🔌 API Integration

Frontend expect backend API tại `http://localhost:8000` với các endpoints:

### POST `/api/detect/image`
Upload ảnh và nhận kết quả phát hiện.

**Request:**
```javascript
FormData {
  file: <image file>
}
```

**Response:**
```json
{
  "detections": [
    {
      "class": "stop",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 120,
        "width": 150,
        "height": 150
      }
    }
  ],
  "annotated_image": "<base64 encoded image>"
}
```

### POST `/api/detect/video`
Upload video và nhận video đã được xử lý.

**Request:**
```javascript
FormData {
  file: <video file>
}
```

**Response:**
```javascript
// Video file (blob)
```

## 🎯 Proxy Configuration

Vite được config để proxy API requests từ `/api/*` sang `http://localhost:8000`. Xem `vite.config.js`.

## 🎨 Customization

### Tailwind Theme
Chỉnh sửa `tailwind.config.js` để thay đổi theme colors, fonts, etc.

### Component Styles
Chỉnh sửa classes Tailwind trong từng component.

## 🐛 Troubleshooting

### Backend không connect được?
- Kiểm tra backend đang chạy tại `http://localhost:8000`
- Kiểm tra proxy config trong `vite.config.js`

### Build lỗi?
- Xóa `node_modules` và `dist`, sau đó `npm install` lại
- Kiểm tra version Node.js (khuyến nghị >= 16)

