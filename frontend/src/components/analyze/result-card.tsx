"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

type ResultCardProps = {
  result: {
    model: string;
    score: number;
    confidence: number;
  };
  index: number;
  onFeedback: (isCorrect: boolean) => void;
};

export function ResultCard({ result, index, onFeedback }: ResultCardProps) {
  const [feedbackGiven, setFeedbackGiven] = useState<boolean | null>(null);
  
  const handleFeedback = (isCorrect: boolean) => {
    setFeedbackGiven(isCorrect);
    onFeedback(isCorrect);
  };

  const getColor = (score: number) => {
    if (score >= 0.8) return 'text-red-600';
    if (score >= 0.6) return 'text-orange-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getProgressBarColor = (score: number) => {
    if (score >= 0.8) return 'bg-red-600';
    if (score >= 0.6) return 'bg-orange-600';
    if (score >= 0.4) return 'bg-yellow-600';
    return 'bg-green-600';
  };

  // Convert confidence to percentage for display
  const confidencePercentage = Math.round(result.confidence * 100);
  const scorePercentage = Math.round(result.score * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index, duration: 0.5 }}
      className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 hover:shadow-lg transition-shadow"
    >
      <div className="p-5"> 
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">{result.model}</h3>
          <span className={`font-bold text-lg ${getColor(result.score)}`}>
            {scorePercentage}%
          </span>
        </div>
        
        <div className="mb-4">
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(result.score)}`}
              style={{ width: `${scorePercentage}%` }}
            ></div>
          </div>
          <div className="flex justify-between mt-1 text-xs text-gray-500">
            <span>Humain</span>
            <span>IA</span>
          </div>
        </div>
        
        <div className="text-sm text-gray-600 mb-4">
          <p>
            Ce texte a {scorePercentage}% de chances d'avoir été généré par l'IA
            avec une confiance de <strong>{confidencePercentage}%</strong>.
          </p>
        </div>

        <div className="border-t pt-4">
          <p className="text-sm text-gray-600 mb-2">Ce résultat vous semble-t-il correct ?</p>
          <div className="flex gap-4">
            <button
              onClick={() => handleFeedback(true)}
              disabled={feedbackGiven !== null}
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                feedbackGiven === true
                  ? 'bg-green-100 text-green-700'
                  : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              <ThumbsUp size={14} />
              <span>Oui</span>
            </button>
            <button
              onClick={() => handleFeedback(false)}
              disabled={feedbackGiven !== null}
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                feedbackGiven === false
                  ? 'bg-red-100 text-red-700'
                  : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              <ThumbsDown size={14} />
              <span>Non</span>
            </button>
          </div>
          
          {feedbackGiven !== null && (
            <p className="text-xs text-green-600 mt-2">
              Merci pour votre retour ! Il nous aidera à améliorer notre système.
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}
