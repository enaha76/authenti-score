import { MainLayout } from '@/components/layouts/main-layout';
import { MotionWrapper } from "@/components/animations/motion-wrapper";
import Image from 'next/image';

export default function AboutPage() {
  return (
    <MainLayout>
      <section className="py-12 md:py-20">
        <div className="container max-w-4xl mx-auto px-4">
          <MotionWrapper
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-4">
              À propos d'<span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">AuthentiScore</span>
            </h1>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Notre mission, notre approche et notre technologie pour détecter le contenu généré par l'IA.
            </p>
          </MotionWrapper>
          
          <div className="space-y-16">
            <MotionWrapper
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="flex flex-col md:flex-row gap-8 items-center"
            >
              <div className="md:w-1/2">
                <h2 className="text-2xl font-bold mb-4">Notre Mission</h2>
                <p className="text-gray-700 mb-4">
                  À l'ère du contenu généralisé créé par l'IA, il devient de plus en plus difficile de distinguer les textes écrits par des humains de ceux produits par des machines. AuthentiScore a été développé pour répondre à ce défi.
                </p>
                <p className="text-gray-700">
                  Notre objectif est de fournir un outil fiable et précis qui permet aux utilisateurs de déterminer la probabilité qu'un texte ait été généré par une intelligence artificielle, contribuant ainsi à maintenir l'authenticité et la transparence dans la communication numérique.
                </p>
              </div>
              <div className="md:w-1/2 flex justify-center">
                <div className="w-64 h-64 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center">
                  <div className="text-6xl">📝</div>
                </div>
              </div>
            </MotionWrapper>

            <MotionWrapper
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="flex flex-col md:flex-row-reverse gap-8 items-center"
            >
              <div className="md:w-1/2">
                <h2 className="text-2xl font-bold mb-4">Notre Technologie</h2>
                <p className="text-gray-700 mb-4">
                  AuthentiScore utilise des modèles de classification avancés entraînés sur une grande variété de textes générés par différents modèles d'IA (GPT-2, GPT-3, LLaMA 2, Claude 2) et des textes écrits par des humains.
                </p>
                <p className="text-gray-700">
                  Notre système analyse diverses caractéristiques linguistiques, patterns de répétition, cohérence sémantique et autres marqueurs subtils qui différencient le contenu généré par l'IA du contenu humain authentique.
                </p>
              </div>
              <div className="md:w-1/2 flex justify-center">
                <div className="w-64 h-64 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center">
                  <div className="text-6xl">🤖</div>
                </div>
              </div>
            </MotionWrapper>

            <MotionWrapper
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
            >
              <h2 className="text-2xl font-bold mb-6 text-center">Notre Équipe</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
                {team.map((member, index) => (
                  <div key={index} className="text-center p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow">
                    <div className="w-24 h-24 rounded-full bg-gradient-to-r from-blue-200 to-purple-200 mx-auto mb-4 flex items-center justify-center text-2xl font-bold text-blue-600">
                      {member.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <h3 className="text-lg font-semibold mb-1">{member.name}</h3>
                    <p className="text-sm text-gray-600">{member.role}</p>
                  </div>
                ))}
              </div>
            </MotionWrapper>
          </div>
        </div>
      </section>
    </MainLayout>
  );
}

const team = [
  { name: 'Cheikh Ahmedou Enaha', role: 'Lead Developer' },
  { name: 'Djilit Abdellahi', role: 'Data Scientist' },
  { name: 'Mohamed Abderhman Nanne', role: 'Frontend Developer' },
  { name: 'Mohamed Lemin Taleb Ahmed', role: 'ML Engineer' },
];
