'use client'

import Image from "next/image";
import Link from 'next/link';
import { MainLayout } from '@/components/layouts/main-layout';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <MainLayout>
      <section className="flex flex-col items-center justify-center gap-8 py-24 md:py-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center space-y-4"
        >
          <h1 className="text-4xl font-bold tracking-tight text-black sm:text-6xl">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Authenti</span>
            Score
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Détectez facilement si un texte a été généré par une intelligence artificielle.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="w-full max-w-4xl"
        >
          <div className="bg-white p-6 rounded-2xl shadow-xl border border-gray-100">
            <div className="w-full h-64 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg flex items-center justify-center mb-6">
              <div className="text-6xl">🔍</div>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600">
                <Link
                  href="/analyze"
                  className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto"
                >
                  Analyser un texte
                </Link>
              </Button>
              <Button asChild size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600">
                <Link
                  href="/analyze-image"
                  className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto"
                >
                  Analyser une image
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <Link href="/about">En savoir plus</Link>
              </Button>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl w-full mt-12"
        >
          {features.map((feature, index) => (
            <div 
              key={index}
              className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow"
            >
              <div className="text-blue-600 mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </motion.div>
      </section>
    </MainLayout>
  );
}

const features = [
  {
    title: 'Détection Précise',
    description: 'Algorithmes avancés capables de détecter les textes générés par GPT-2, GPT-3, LLaMA 2, et Claude 2.',
    icon: <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">🔍</div>,
  },
  {
    title: 'Feedback Utilisateur',
    description: 'Contribuez à améliorer le système en validant ou corrigeant les résultats d\'analyse.',
    icon: <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">👥</div>,
  },
  {
    title: 'Tableau de Performance',
    description: 'Consultez les métriques de performance pour évaluer la précision et la qualité du modèle.',
    icon: <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">📊</div>,
  },
];
