import { useState, useRef } from 'react'
import axios from 'axios'

const VideoDetector = ({ onResult }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return

    if (!selectedFile.type.startsWith('video/')) {
      setError('Vui lòng chọn file video (mp4, avi, mov)')
      return
    }

    setFile(selectedFile)
    setError(null)

    // Tạo preview video
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result)
    }
    reader.readAsDataURL(selectedFile)
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Vui lòng chọn video trước')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Gửi video đến API
      const response = await axios.post('/api/detect/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', // Nhận video đã xử lý
      })

      // Tạo URL để hiển thị video kết quả
      const videoUrl = URL.createObjectURL(response.data)
      
      onResult({
        type: 'video',
        video: videoUrl,
        originalVideo: preview,
        fileName: file.name,
      })
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Không thể kết nối đến server. Vui lòng kiểm tra backend!'
      )
      console.error('Error:', err)
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
        🎥 Phát hiện biển báo trên video
      </h2>

      {/* File Input */}
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="hidden"
          id="video-upload"
        />
        <label
          htmlFor="video-upload"
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
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <span className="text-gray-600 font-medium">
            Click để chọn video hoặc kéo thả
          </span>
          <span className="text-sm text-gray-500 mt-1">
            MP4, AVI, MOV (tối đa 50MB)
          </span>
        </label>
      </div>

      {/* Preview */}
      {preview && (
        <div className="relative">
          <video
            src={preview}
            controls
            className="w-full h-auto rounded-lg shadow-md"
          />
          <div className="mt-2 text-sm text-gray-600">
            File: {file?.name} ({(file?.size / 1024 / 1024).toFixed(2)} MB)
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
              : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-xl'
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
            '🎬 Phát hiện'
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

      {/* Warning */}
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded text-sm">
        ⚠️ Xử lý video có thể mất nhiều thời gian tùy vào độ dài và chất lượng video.
      </div>
    </div>
  )
}

export default VideoDetector

