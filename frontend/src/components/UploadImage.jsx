import { useState, useRef } from 'react'
import axios from 'axios'

const UploadImage = ({ onResult }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return

    if (!selectedFile.type.startsWith('image/')) {
      setError('Vui lòng chọn file ảnh (jpg, png, jpeg)')
      return
    }

    setFile(selectedFile)
    setError(null)

    // Tạo preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result)
    }
    reader.readAsDataURL(selectedFile)
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Vui lòng chọn ảnh trước')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/detect/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      const data = response?.data || {}
      const simple = Array.isArray(data.objects_simple) ? data.objects_simple : []
      const mapped = (simple.length > 0 ? simple : (data.detections || [])).map(d => {
        if (Array.isArray(d.bbox)) {
          const x1 = d.bbox[0] ?? 0, y1 = d.bbox[1] ?? 0, x2 = d.bbox[2] ?? 0, y2 = d.bbox[3] ?? 0
          return {
            class: d.label || d.cls_name,
            confidence: d.confidence ?? d.conf,
            bbox: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
          }
        }
        return {
          class: d.label || d.cls_name,
          confidence: d.confidence ?? d.conf ?? d.score,
          bbox: {
            x: d.box?.x ?? 0,
            y: d.box?.y ?? 0,
            width: (d.box?.w ?? 0),
            height: (d.box?.h ?? 0)
          }
        }
      })

      onResult({
        type: 'image',
        image: preview,
        detections: mapped,
        annotatedImage: data?.annotated_image || null,
      })
    } catch (err) {
      console.error('Full error:', err)
      console.error('Error response:', err.response)
      setError(
        err.response?.data?.detail || 
        `Không thể kết nối đến server. Lỗi: ${err.message}`
      )
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setError(null)
    setLoading(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    onResult(null)
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        📷 Phát hiện biển báo trên ảnh
      </h2>

      {/* File Input */}
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <svg
            className="w-16 h-16 text-gray-400 mb-3"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <span className="text-gray-600 font-medium">
            Click để chọn ảnh hoặc kéo thả
          </span>
          <span className="text-sm text-gray-500 mt-1">
            JPG, PNG, JPEG (tối đa 10MB)
          </span>
        </label>
      </div>

      {/* Preview */}
      {preview && (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-auto rounded-lg shadow-md"
          />
          <div className="mt-2 text-sm text-gray-600">
            File: {file?.name}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Buttons */}
      <div className="flex space-x-3">
        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            loading || !file
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Đang xử lý...
            </span>
          ) : (
            '🚀 Phát hiện'
          )}
        </button>
        {file && (
          <button
            onClick={handleReset}
            className="px-6 py-3 rounded-lg font-semibold bg-gray-200 hover:bg-gray-300 text-gray-700 transition-all"
          >
            🔄 Đặt lại
          </button>
        )}
      </div>
    </div>
  )
}

export default UploadImage

