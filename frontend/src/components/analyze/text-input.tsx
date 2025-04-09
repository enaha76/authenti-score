interface TextInputProps {
  text: string;
  setText: (text: string) => void;
  error: string | null;
}

export function TextInput({ text, setText, error }: TextInputProps) {
  return (
    <div className="space-y-2">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Collez votre texte ici pour l'analyser..."
        className="w-full h-48 p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none font-[family-name:var(--font-geist-sans)]"
      />
      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}
    </div>
  );
}
