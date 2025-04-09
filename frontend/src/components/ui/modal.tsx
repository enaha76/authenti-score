"use client";

import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

export function Modal({ isOpen, onClose, children }: ModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ 
              opacity: 1, 
              scale: 1, 
              y: 0,
              transition: {
                type: "spring",
                stiffness: 300,
                damping: 25
              }
            }}
            exit={{ 
              opacity: 0, 
              scale: 0.95, 
              y: -20,
              transition: {
                duration: 0.2
              }
            }}
            className="fixed inset-0 flex items-center justify-center p-4 z-50"
          >
            <motion.div
              className="w-full max-w-2xl bg-white rounded-lg border border-gray-100 shadow-2xl transition-all duration-200"
            >
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={onClose}
              className="absolute right-4 top-4 text-gray-500 hover:text-gray-700 transition-colors"
            >
              <X size={24} />
            </motion.button>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ 
                opacity: 1, 
                y: 0,
                transition: {
                  delay: 0.1,
                  duration: 0.2
                }
              }}
              className="p-6"
            >
              {children}
            </motion.div>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
