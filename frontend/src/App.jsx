import { useState, useCallback, useEffect } from 'react'
import './App.css'

const API = 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [searched, setSearched] = useState(false)
  const [lightbox, setLightbox] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch(`${API}/api/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(() => {})
  }, [])

  const search = useCallback(async () => {
    const trimmed = query.trim()
    if (!trimmed) return

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const res = await fetch(
        `${API}/api/search?q=${encodeURIComponent(trimmed)}&n=40`
      )
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError(err.message || 'Search failed. Is the backend running?')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [query])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') search()
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Image Search</h1>
        <p>Search your photos using natural language</p>
      </header>

      <div className="search-container">
        <div className="search-box">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you're looking for... e.g. dog at the park"
            autoFocus
          />
          <button onClick={search} disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {searched && !loading && results.length > 0 && (
        <div className="stats-bar">
          <span>{results.length} results for "{query}"</span>
          {stats && <span>{stats.indexed_images} images indexed</span>}
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Searching your photos...</p>
        </div>
      )}

      {!loading && searched && results.length === 0 && !error && (
        <div className="empty-state">
          <div className="icon">🔍</div>
          <p>No results found. Try a different description.</p>
        </div>
      )}

      {!searched && !loading && (
        <div className="empty-state">
          <div className="icon">🖼️</div>
          <p>Type a description to search your photos</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
            Try: "sunset", "people smiling", "mountains", "food"
          </p>
        </div>
      )}

      {results.length > 0 && (
        <div className="results-grid">
          {results.map((item) => (
            <div
              key={item.id}
              className="image-card"
              onClick={() => setLightbox(item)}
            >
              <img
                src={`${API}${item.url}`}
                alt={item.filename}
                loading="lazy"
              />
              <div className="card-info">
                <div className="filename" title={item.filename}>
                  {item.filename}
                </div>
                <div className="score">
                  {(item.score * 100).toFixed(1)}% match
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <button
            className="lightbox-close"
            onClick={() => setLightbox(null)}
            aria-label="Close"
          >
            ✕
          </button>
          <img
            src={`${API}${lightbox.url}`}
            alt={lightbox.filename}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  )
}

export default App
