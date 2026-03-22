PATTERNS = {
    "suin_juriscol_ui": [
        # Navigation buttons and UI elements
        r"Ir al portal SUIN-Juriscol",
        r"Ayúdanos a mejorar",
        r"Guardar en PDF o imprimir la norma",
        r"Responder Encuesta",
        r"Descargar en Word",
        r"Compartir este documento",
        r"Lectura de voz",
        
        # Index and section headers
        r"I\s*N\s*D\s*I\s*C\s*E\s*\[Mostrar\]",
        r"R\s*E\s*S\s*U\s*M\s*E\s*N\s+D\s*E\s+M\s*O\s*D\s*I\s*F\s*I\s*C\s*A\s*C\s*I\s*O\s*N\s*E\s*S\s*\[Mostrar\]",
        r"R\s*E\s*S\s*U\s*M\s*E\s*N\s+D\s*E\s+J\s*U\s*R\s*I\s*S\s*P\s*R\s*U\s*D\s*E\s*N\s*C\s*I\s*A\s*\[Mostrar\]",
        r"T\s*E\s*X\s*T\s*O\s+C\s*O\s*R\s*R\s*E\s*S\s*P\s*O\s*N\s*D\s*I\s*E\s*N\s*T\s*E\s+A\s*\[Mostrar\]",
        r"J\s*U\s*R\s*I\s*S\s*P\s*R\s*U\s*D\s*E\s*N\s*C\s*I\s*A\s*\[Mostrar\]",
        r"L\s*E\s*G\s*I\s*S\s*L\s*A\s*C\s*I\s*[OÓ]\s*N\s+A\s*N\s*T\s*E\s*R\s*I\s*O\s*R\s*\[Mostrar\]",
        
        # Course and promotional content
        r"^\s*Curso\s+SUIN-Juriscol\s*$",
        r".*Curso\s+SUIN-Juriscol.*",
        r".*Curso\s+caracter[íi]sticas.*",
        r"de constitucionalidad y nulidad\s*$",
        r"^.*Inscripciones abiertas.*$",
        r"^\s*Curso\s+Calidad\s+Normativa\s*$",
        r".*Curso\s+Calidad\s+Normativa.*",
        
        # General UI elements
        r"\[Mostrar\]",
    ],
    "urls": [
        
        # URLs and web addresses
        r"https?://\S+",
        r"www\.\S+",
    ],
    "editorial_metadata": [
        
        # Editorial metadata and publication information
        r"DIARIO OFICIAL.*",
        r"Diario Oficial.*",
        r"AÑO\s+[CLXIV]+\s+No\.\s*\d+.*",
        r"Edición de\s+\d+\s+páginas.*",
        r"PAG\.?\s*\d+",
        r"E\s*S\s*T\s*A\s*D\s*O\s+D\s*E\s*V\s*I\s*G\s*E\s*N\s*C\s*I\s*A\s*:.*",
        r"Los datos publicados en SUIN-Juriscol.*",
    ],
    "navigation_timestamps": [
        
        # Navigation timestamps and date-time patterns
        r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}.*",
        r"\b\d+/\d+\b\s*$",
    ],
    "page_numbers": [
        
        # Page numbers and pagination patterns
        r"^\s*\d+\s*$",
        r"PÁGINA\s+\d+(\s+DE\s+\d+)?",
        r"Página\s+\d+\s+de\s+\d+",
    ],
    "headers_footers": [
        
        # Headers, footers, and common document elements
        r"^[-_=]+\s*$",
        r"^\s*©.*$",
        r"^\s*confidential.*$",
        r"El Presidente del Honorable.*",
        r"El Secretario General del Honorable.*",
    ],
}
