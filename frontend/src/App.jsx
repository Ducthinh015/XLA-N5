import { useState } from 'react'
import UploadImage from './components/UploadImage'
import ResultDisplay from './components/ResultDisplay'
import VideoDetector from './components/VideoDetector'

function App() {
  const [activeTab, setActiveTab] = useState('image')
  const [detectionResult, setDetectionResult] = useState(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-800">
            🚦 Nhận diện biển báo giao thông
          </h1>
          <p className="text-gray-600 mt-2">
            Phát hiện biển báo bằng AI YOLO
          </p>
        </div>
      </header>

      {/* Tabs */}
      <div className="container mx-auto px-4 py-6">
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setActiveTab('image')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'image'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-blue-50'
            }`}
          >
            📷 Phát hiện ảnh
          </button>
          <button
            onClick={() => setActiveTab('video')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'video'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-blue-50'
            }`}
          >
            🎥 Phát hiện video
          </button>
        </div>

        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            {activeTab === 'image' ? (
              <UploadImage onResult={setDetectionResult} />
            ) : (
              <VideoDetector onResult={setDetectionResult} />
            )}
          </div>

          {/* Result Display */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <ResultDisplay result={detectionResult} type={activeTab} />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-gray-600">
        <p>XLA - Nhận diện biển báo giao thông © 2025</p>
      </footer>
    </div>
  )
}

export default App

