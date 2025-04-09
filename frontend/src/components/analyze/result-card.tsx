"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { AnalysisResult } from '@/app/analyze/page';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

type ResultCardProps = {
  result: AnalysisResult;
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
    if (score > 0.8) return 'text-red-600';
    if (score > 0.6) return 'text-orange-600';
    if (score > 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

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
            {Math.round(result.score * 100)}%
          </span>
        </div>
        
        <div className="mb-4">
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full ${getColor(result.score).replace('text-', 'bg-')}`}
              style={{ width: `${result.score * 100}%` }}
            ></div>
          </div>
          <div className="flex justify-between mt-1 text-xs text-gray-500">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
        
        <div className="text-sm text-gray-600 mb-4">
          <p>
            Probabilité que ce texte ait été généré par le modèle <strong>{result.model}</strong> avec une confiance de <strong>{Math.round(result.confidence * 100)}%</strong>.
          </p>
        </div>
        
        <div className="border-t pt-3 mt-3">
          <p className="text-sm text-gray-700 mb-2">Ce résultat vous semble-t-il correct ?</p>
          <div className="flex space-x-2">
            <button 
              onClick={() => handleFeedback(true)}
              disabled={feedbackGiven !== null}
              className={`flex items-center space-x-1 px-3 py-1 rounded text-sm ${feedbackGiven === true ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'} transition-colors`}
            >
              <ThumbsUp size={14} />
              <span>Oui</span>
            </button>
            <button 
              onClick={() => handleFeedback(false)}
              disabled={feedbackGiven !== null}
              className={`flex items-center space-x-1 px-3 py-1 rounded text-sm ${feedbackGiven === false ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'} transition-colors`}
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
