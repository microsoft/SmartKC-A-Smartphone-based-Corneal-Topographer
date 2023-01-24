package com.example.kt.data.interceptors

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.stringPreferencesKey
import com.example.kt.BuildConfig
import com.example.kt.utils.PreferenceKeys
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.Response

class HostSelectionInterceptor(val dataStore: DataStore<Preferences>) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val newRequestBuilder = request.newBuilder()
        val newRequest = runBlocking {
            val data = dataStore.data.first()
            val uploadUrlKey = stringPreferencesKey(PreferenceKeys.UPLOAD_URL)
            val url = (data[uploadUrlKey] ?: BuildConfig.UPLOAD_URL).trimEnd('/')
            val newUrl = request.url.toString().replace("http://__url__", url)
            newRequestBuilder
                .url(newUrl)
                .build()
        }
        return chain.proceed(newRequest)
    }
}