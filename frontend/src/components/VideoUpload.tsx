'use client';
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Analysis {
  timestamp: number;
  analysis: string;
}

export default function VideoUpload() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleVideoUpload = async (file: File) => {
    setIsAnalyzing(true);
    setAnalyses([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze-video', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setAnalyses(result.analyses);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files[0] && files[0].type.startsWith('video/')) {
      handleVideoUpload(files[0]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isAnalyzing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            Always-On Fitness Coach {isAnalyzing && '- Analyzing...'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="video/*"
              onChange={(e) => e.target.files?.[0] && handleVideoUpload(e.target.files[0])}
              className="hidden"
              id="video-upload"
            />
            <label htmlFor="video-upload" className="cursor-pointer">
              <div className="text-xl mb-2">🎥</div>
              <p className="text-lg font-medium">Upload Squat Video</p>
              <p className="text-gray-600">Drag & drop or click to select</p>
            </label>
          </div>
        </CardContent>
      </Card>

      {analyses.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Form Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analyses.map((analysis, index) => (
                <div key={index} className="border-l-4 border-blue-500 pl-4">
                  <div className="text-sm text-gray-600">
                    {analysis.timestamp.toFixed(1)}s
                  </div>
                  <div className="text-gray-800">{analysis.analysis}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
