import Link from "next/link";

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-100 py-8">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <Link href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Authenti</span>
              <span className="text-xl font-bold">Score</span>
            </Link>
            <p className="text-gray-500 mt-2">
              © {new Date().getFullYear()} AuthentiScore. Tous droits réservés.
            </p>
          </div>
          
          <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-8">
            <div>
              <h3 className="font-semibold mb-2 text-gray-800">Navigation</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="/" className="text-gray-500 hover:text-blue-600 transition-colors">
                    Accueil
                  </Link>
                </li>
                <li>
                  <Link href="/analyze" className="text-gray-500 hover:text-blue-600 transition-colors">
                    Analyser
                  </Link>
                </li>
                <li>
                  <Link href="/about" className="text-gray-500 hover:text-blue-600 transition-colors">
                    À propos
                  </Link>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2 text-gray-800">Contact</h3>
              <ul className="space-y-2">
                <li className="text-gray-500">
                  <a href="mailto:contact@authentiscore.com" className="hover:text-blue-600 transition-colors">
                    contact@authentiscore.com
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
