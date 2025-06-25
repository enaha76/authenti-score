"use client";

import { useEffect } from 'react';

interface ImageInputProps {
  file: File | null;
  setFile: (file: File | null) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
}

export function ImageInput({ file, setFile, previewUrl, setPreviewUrl }: ImageInputProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files && e.target.files[0];
    if (selected) {
      setFile(selected);
      const url = URL.createObjectURL(selected);
      setPreviewUrl(url);
    } else {
      setFile(null);
      setPreviewUrl(null);
    }
  };

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="space-y-2">
      {previewUrl && (
        <img src={previewUrl} alt="Preview" className="max-h-60 mx-auto rounded-lg" />
      )}
      <input type="file" accept="image/*" onChange={handleChange} />
    </div>
  );
}
