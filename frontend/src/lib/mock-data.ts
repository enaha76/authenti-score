export interface AnalysisResult {
  model: string;
  score: number;
  confidence: number;
}

export const mockResults: AnalysisResult[] = [
  {
    model: "GPT-3.5",
    score: 0.89,
    confidence: 0.95
  },
  {
    model: "Claude 2",
    score: 0.45,
    confidence: 0.88
  },
  {
    model: "LLaMA 2",
    score: 0.72,
    confidence: 0.91
  },
  {
    model: "GPT-4",
    score: 0.93,
    confidence: 0.97
  }
];