package com.example.kt.injection

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.stringPreferencesKey
import com.example.kt.BuildConfig
import com.example.kt.service.FileAPI
import com.example.kt.utils.PreferenceKeys
import dagger.Module
import dagger.Provides
import dagger.Reusable
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class RetrofitModule {

    @Provides
    @Reusable
    fun provideOkHttp(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.MINUTES)
            .readTimeout(10, TimeUnit.MINUTES)
            .build()
    }

    @Provides
    fun provideRetrofitInstance(
        dataStore: DataStore<Preferences>,
        okHttpClient: OkHttpClient
    ): Retrofit {
        var url: String
        runBlocking {
            val data = dataStore.data.first()
            val uploadUrlKey = stringPreferencesKey(PreferenceKeys.UPLOAD_URL)
            url = data[uploadUrlKey] ?: BuildConfig.UPLOAD_URL
        }
        return Retrofit.Builder().client(okHttpClient).baseUrl(url).build()
    }

    @Provides
    @Reusable
    fun provideLanguageAPI(retrofit: Retrofit): FileAPI {
        return retrofit.create(FileAPI::class.java)
    }
}