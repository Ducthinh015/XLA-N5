const ResultDisplay = ({ result, type }) => {
  if (!result) {
    return (
      <div className="flex items-center justify-center h-full min-h-[300px]">
        <div className="text-center text-gray-500">
          <svg
            className="w-24 h-24 mx-auto mb-4 text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p className="text-lg font-medium">
            Kết quả phát hiện sẽ hiển thị ở đây
          </p>
          <p className="text-sm mt-2">
            Upload ảnh hoặc video để bắt đầu
          </p>
        </div>
      </div>
    )
  }

  // Image Result
  if (type === 'image' && result.detections) {
    const detections = result.detections
    
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          ✅ Kết quả phát hiện
        </h2>

        {/* Annotated Image */}
        {result.annotatedImage && (
          <div className="mb-4">
            <img
              src={`data:image/jpeg;base64,${result.annotatedImage}`}
              alt="Result"
              className="w-full h-auto rounded-lg shadow-md border-2 border-blue-500"
            />
          </div>
        )}

        {/* Statistics */}
        <div className="bg-blue-50 rounded-lg p-4 mb-4">
          <h3 className="font-semibold text-gray-800 mb-2">
            📊 Thống kê
          </h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-600">Tổng số biển báo:</span>
              <span className="font-bold text-blue-600 ml-2">
                {detections.length}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Độ tin cậy TB:</span>
              <span className="font-bold text-green-600 ml-2">
                {detections.length > 0
                  ? (detections.reduce((sum, d) => sum + d.confidence, 0) /
                      detections.length *
                      100).toFixed(1)
                  : 0}
                %
              </span>
            </div>
          </div>
        </div>

        {/* Detections List */}
        {detections.length > 0 && (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="bg-gray-50 px-4 py-2 font-semibold text-gray-700">
              🔍 Chi tiết phát hiện
            </div>
            <div className="max-h-[300px] overflow-y-auto">
              {detections.map((detection, index) => (
                <div
                  key={index}
                  className="border-b border-gray-100 px-4 py-3 hover:bg-blue-50 transition-colors"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-semibold text-gray-800">
                        {detection.class}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Vị trí: ({detection.bbox.x.toFixed(0)}, {detection.bbox.y.toFixed(0)}) - 
                        Kích thước: {detection.bbox.width.toFixed(0)} x {detection.bbox.height.toFixed(0)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                        detection.confidence > 0.7
                          ? 'bg-green-100 text-green-800'
                          : detection.confidence > 0.5
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {(detection.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {detections.length === 0 && (
          <div className="bg-gray-50 rounded-lg p-6 text-center text-gray-500">
            <svg
              className="w-16 h-16 mx-auto mb-2 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            Không phát hiện biển báo nào
          </div>
        )}
      </div>
    )
  }

  // Video Result
  if (type === 'video' && result.video) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          ✅ Video đã xử lý
        </h2>

        {/* Video Player */}
        <div className="relative">
          <video
            src={result.video}
            controls
            autoPlay
            className="w-full h-auto rounded-lg shadow-md border-2 border-green-500"
          />
          <div className="mt-2 text-sm text-gray-600 bg-green-50 p-3 rounded rounded-lg">
            📹 Video đã được xử lý và phát hiện biển báo
          </div>
        </div>

        {/* Download Button */}
        <a
          href={result.video}
          download="detected_video.mp4"
          className="block text-center py-3 px-6 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-all shadow-lg"
        >
          ⬇️ Tải video về máy
        </a>
      </div>
    )
  }

  return null
}

export default ResultDisplay

