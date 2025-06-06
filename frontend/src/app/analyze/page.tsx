"use client";

import { useState } from 'react';
import { MainLayout } from '@/components/layouts/main-layout';
import { TextInput } from '@/components/analyze/text-input';
import { ResultCard } from '@/components/analyze/result-card';
import { LoadingAnimation } from '@/components/analyze/loading-animation';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';
import { motion } from 'framer-motion';

interface PredictionResponse {
  text: string;
  prediction: string;
  is_ai_generated: boolean;
  confidence: number;
}

export default function AnalyzePage() {
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAnalyze = async () => {
    if (text.length < 50) {
      setError('Le texte doit contenir au moins 50 caractères.');
      return;
    }

    setError(null);
    setIsAnalyzing(true);
    setIsModalOpen(true);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze text');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Une erreur est survenue lors de l\'analyse.');
      console.error('Error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setText('');
    setError(null);
    setResults(null);
    setIsAnalyzing(false);
    setIsModalOpen(false);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setResults(null);
    setIsAnalyzing(false);
  };

  const handleFeedback = (isCorrect: boolean) => {
    console.log('Feedback received:', isCorrect);
  };

  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-center mb-2">Analyse de Texte</h1>
          <p className="text-center text-gray-600">
            Collez votre texte ci-dessous pour détecter s'il a été généré par l'IA
          </p>
        </div>
        
        <div className="space-y-6">
          <div className="relative">
            <TextInput 
              text={text} 
              setText={setText} 
              error={error} 
            />
            <div className="flex justify-between items-center mt-2 text-sm text-gray-500">
              <span>{text.length} caractères {text.length < 50 && "(minimum 50)"}</span>
              <button 
                onClick={handleReset}
                className="text-blue-600 hover:text-blue-800 transition-colors"
                disabled={text.length === 0}
              >
                Réinitialiser
              </button>
            </div>
          </div>

          <div className="flex justify-center gap-4">
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || text.length < 50}
              className="w-full max-w-xs"
            >
              {isAnalyzing ? 'Analyse en cours...' : 'Analyser le texte'}
            </Button>
          </div>
        </div>

        <Modal isOpen={isModalOpen} onClose={handleCloseModal}>
          <div className="space-y-6">
            <motion.h2 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-2xl font-bold text-center"
            >
              Résultats de l'analyse
            </motion.h2>
            
            {isAnalyzing && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="py-8"
              >
                <LoadingAnimation />
              </motion.div>
            )}

            {results && !isAnalyzing && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="grid gap-4"
              >
                <motion.div
                  key="prediction-result"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ 
                    opacity: 1, 
                    x: 0,
                    transition: {
                      type: "spring",
                      stiffness: 100
                    }
                  }}
                >
                  <ResultCard
                    result={{
                      model: results.prediction,
                      score: results.confidence,
                      confidence: results.confidence
                    }}
                    index={0}
                    onFeedback={handleFeedback}
                  />
                </motion.div>
              </motion.div>
            )}

            {!isAnalyzing && results && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="flex justify-center mt-6"
              >
                <Button
                  onClick={handleCloseModal}
                  variant="outline"
                  className="w-full max-w-xs"
                >
                  Fermer
                </Button>
              </motion.div>
            )}
          </div>
        </Modal>
      </div>
    </MainLayout>
  );
}
