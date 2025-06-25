"use client";

import { useState } from 'react';
import { MainLayout } from '@/components/layouts/main-layout';
import { ImageInput } from '@/components/image/image-input';
import { LoadingAnimation } from '@/components/analyze/loading-animation';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';

interface ImageResponse {
  prediction: string;
  is_ai_generated?: boolean;
  confidence?: number;
  generator?: string;
}

export default function AnalyzeImagePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<ImageResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAnalyze = async () => {
    if (!file) {
      setError("Veuillez sélectionner une image.");
      return;
    }
    setError(null);
    setIsAnalyzing(true);
    setIsModalOpen(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict-image', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Request failed');
      }
      const data = await response.json();
      setResults(data);
    } catch (e) {
      setError("Une erreur est survenue lors de l'analyse.");
      console.error(e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResults(null);
    setIsAnalyzing(false);
    setIsModalOpen(false);
    setError(null);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setResults(null);
    setIsAnalyzing(false);
  };

  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold mb-2">Analyse d'image</h1>
          <p className="text-gray-600">Téléversez une image pour détecter si elle a été générée par l'IA</p>
        </div>

        <div className="space-y-6">
          <ImageInput file={file} setFile={setFile} previewUrl={previewUrl} setPreviewUrl={setPreviewUrl} />
          {error && <p className="text-red-500 text-sm">{error}</p>}
          <div className="flex justify-center gap-4">
            <Button onClick={handleAnalyze} disabled={isAnalyzing || !file} className="w-full max-w-xs">
              {isAnalyzing ? "Analyse en cours..." : "Analyser l'image"}
            </Button>
            <Button onClick={handleReset} variant="outline" className="w-full max-w-xs" disabled={!file && !previewUrl}>
              Réinitialiser
            </Button>
          </div>
        </div>

        <Modal isOpen={isModalOpen} onClose={handleCloseModal}>
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-center">Résultats</h2>
            {isAnalyzing && (
              <div className="py-8">
                <LoadingAnimation />
              </div>
            )}
            {results && !isAnalyzing && (
              <div className="space-y-4 text-center">
                <p className="text-lg font-semibold">{results.prediction}</p>
                {typeof results.confidence !== 'undefined' && (
                  <p className="text-gray-600">Confiance : {Math.round(results.confidence * 100)}%</p>
                )}
                {results.generator && (
                  <p className="text-gray-600">Source générée : {results.generator}</p>
                )}
              </div>
            )}
            {!isAnalyzing && results && (
              <div className="flex justify-center mt-6">
                <Button onClick={handleCloseModal} variant="outline" className="w-full max-w-xs">
                  Fermer
                </Button>
              </div>
            )}
          </div>
        </Modal>
      </div>
    </MainLayout>
  );
}
